import re, sys, statistics as st
cur = None; data = {}   # data[(cfg, kind)][variant] = [per-round medians]
for line in open(sys.argv[1]):
    m = re.match(r"=== ROUND (\d+) VARIANT (\w+)", line)
    if m: cur = (int(m.group(1)), m.group(2)); continue
    m = re.search(r"B=8 T=4096 (\d+/\d+)\s+fwd median ([\d.]+) ms.*bwd median ([\d.]+) ms", line)
    if m and cur:
        for kind, val in (("fwd", m.group(2)), ("bwd", m.group(3))):
            data.setdefault((m.group(1), kind), {}).setdefault(cur[1], {})[cur[0]] = float(val)
for (cfg, kind), d in sorted(data.items()):
    rounds = sorted(set(d.get("fix", {})) & set(d.get("nofix", {})))
    diffs = [100.0 * (d["fix"][r] - d["nofix"][r]) / d["nofix"][r] for r in rounds]
    if len(diffs) < 2: print(cfg, kind, "not enough rounds"); continue
    mu, sd = st.mean(diffs), st.stdev(diffs)
    t = mu / (sd / len(diffs) ** 0.5) if sd > 0 else float("inf")
    print(f"{cfg} {kind}{' (null: fix does not touch it)' if kind == 'fwd' else ''}: nofix median-of-medians "
          f"{st.median(d['nofix'].values()):.4f} ms, fix {st.median(d['fix'].values()):.4f} ms | paired diff "
          f"{mu:+.3f}% (sd {sd:.3f}%, n={len(diffs)}, t={t:+.2f})")
