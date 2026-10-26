#!/usr/bin/env python3
# 2026-10-17: summary of run_item3.sh (item 3: the loss gradient written over the logits + the double allocation removed).
# Standalone (test_loss_grad_inplace.cu): loss / gradient checksums must be equal in all arms; allocator peaks per arm.
# Trainer (bluscript_sft old binary / new switch off / new switch on, twice): [MEM] peaks, step losses, grad norms.
# usage: python3 item3_summary.py <item3_1017 folder>
import os, re, sys

L = sys.argv[1]
LOGIT_MIB = 32768 * 50304 * 2 / 2**20   # one logits-sized bf16 tensor at 8 x 4096 x 50,304 = 3,144 MiB

print("STANDALONE (lm_head + loss through autograd, 8 x 4096 x 50,304 bf16, real stage-0 targets)")
res = {}
for arm in ("old_out", "new_out", "new_in"):
    p = f"{L}/test_{arm}.log"
    if not os.path.exists(p):
        print(f"  {arm}: no log"); continue
    m = [l for l in open(p) if l.startswith("RESULT")]
    if not m:
        print(f"  {arm}: no RESULT line; last lines: {open(p).read().splitlines()[-3:]}"); continue
    res[arm] = dict(kv.split("=", 1) for kv in m[0].split()[1:])
if res:
    print(f"  {'arm':8s} {'loss':>10s} {'W.grad checksum':>17s} {'X.grad checksum':>17s} {'reps same':>9s} {'bwd ms':>8s} "
          f"{'allocated_peak':>15s} {'vs old':>9s} {'active_peak':>12s} {'reserved_peak':>14s}")
    base = float(res["old_out"]["allocated_peak_MiB"]) if "old_out" in res else None
    for arm, r in res.items():
        ap = float(r["allocated_peak_MiB"])
        d = f"{ap - base:+9.1f}" if base is not None else ""
        print(f"  {arm:8s} {r['loss']:>10s} {r['wgrad_cs']:>17s} {r['xgrad_cs']:>17s} {r['reps_identical']:>9s} {r['bwd_ms']:>8s} "
              f"{ap:15.1f} {d} {float(r['active_peak_MiB']):12.1f} {float(r['reserved_peak_MiB']):14.1f}")
    same = len({(r["loss"], r["wgrad_cs"], r["xgrad_cs"]) for r in res.values()}) == 1
    print(f"  gradients: {'IDENTICAL in all arms' if same else 'DIFFERENT between arms'} | one logits-sized tensor = {LOGIT_MIB:.1f} MiB")

print("\nTRAINER (bluscript_sft, tiny stage-0 set: 64 train / 16 val rows, B 8, BLU_SFT_EPOCHS=4 = 4 steps, random init)")
rows = []
for arm in ("old_r1", "off_r1", "on_r1", "old_r2", "off_r2", "on_r2"):
    p = f"{L}/trainer_{arm}/run.log"
    if not os.path.exists(p):
        continue
    t = open(p).read()
    mem = re.search(r"\[MEM\] active_peak (\d+) MiB \| allocated_peak (\d+) MiB \| reserved_peak (\d+) MiB", t)
    steps = re.findall(r"^step +(\d+) \| train ([0-9.]+) \| val ([0-9.na-]+) .*?\| norm ([0-9.]+) \| dt ([0-9.]+) ms", t, re.M)
    inplace_msg = "BLU_LOSS_GRAD_INPLACE=1" in t
    done = "Training complete" in t
    rows.append((arm, mem.groups() if mem else None, steps, inplace_msg, done))
if rows:
    print(f"  {'run':7s} {'done':>4s} {'switch msg':>10s} {'active_peak':>11s} {'allocated_peak':>14s} {'reserved_peak':>13s} | steps (train loss / norm)")
    for arm, mem, steps, msg, done in rows:
        ms = f"{int(mem[0]):11d} {int(mem[1]):14d} {int(mem[2]):13d}" if mem else f"{'-':>11s} {'-':>14s} {'-':>13s}"
        st = "  ".join(f"s{s[0]} {s[1]}/{s[3]}" for s in steps)
        print(f"  {arm:7s} {'yes' if done else 'NO':>4s} {'yes' if msg else 'no':>10s} {ms} | {st}")
    def peak(kind, i):
        v = [int(m[i]) for a, m, s, x, d in rows if m and a.startswith(kind)]
        return sum(v) / len(v) if v else None
    for i, name in ((1, "allocated_peak"), (0, "active_peak")):
        o, f_, n = peak("old", i), peak("off", i), peak("on", i)
        if None not in (o, f_, n):
            print(f"  {name}: old {o:.0f} -> new-off {f_:.0f} ({f_ - o:+.0f} MiB = {(f_ - o) / LOGIT_MIB:+.2f} logits) -> "
                  f"new-on {n:.0f} ({n - f_:+.0f} MiB = {(n - f_) / LOGIT_MIB:+.2f} logits); total {n - o:+.0f} MiB")
    s0 = {a: s[0][1] for a, m, s, x, d in rows if s}
    print(f"  step-0 train loss per run: {s0}  -> {'all equal' if len(set(s0.values())) == 1 else 'NOT all equal'}")
