# checks a tiny SFT set the way SftDataLoader does: sizes from meta N/T, valid_counts == non-65535 labels per row,
# and that the rows are the first N rows of the full set (plain Python, no numpy)
import array, json, sys
td, src = sys.argv[1], sys.argv[2]
def rd(path, code, n=None):
    a = array.array(code)
    with open(path, "rb") as f:
        data = f.read() if n is None else f.read(n * a.itemsize)
    a.frombytes(data); return a
for s in ("train", "val"):
    m = json.load(open(f"{td}/{s}/meta.json")); N, T = m["N"], m["T"]
    inp, lab, cnt = rd(f"{td}/{s}/inputs.bin", "H"), rd(f"{td}/{s}/labels.bin", "H"), rd(f"{td}/{s}/valid_counts.bin", "I")
    assert len(inp) == N * T and len(lab) == N * T and len(cnt) == N, (s, len(inp), len(lab), len(cnt))
    real = [sum(1 for v in lab[r * T:(r + 1) * T] if v != 65535) for r in range(N)]
    assert real == list(cnt), (s, [r for r in range(N) if real[r] != cnt[r]])
    assert rd(f"{src}/{s}/labels.bin", "H", N * T) == lab and rd(f"{src}/{s}/inputs.bin", "H", N * T) == inp
    print(f"{s}: N {N} T {T} | sizes ok | valid_counts == labels per row | rows identical to the first {N} of the full set"
          f" | trained tokens {sum(cnt):,} (min {min(cnt)} max {max(cnt)}) | max input id {max(inp)}")
