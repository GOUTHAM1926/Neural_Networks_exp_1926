# Two ways to build SFT steps, simulated on the REAL Tülu 3 SFT lengths we trained on (2026-10-24).
#   A (ours, offline pre-packing): the whole dataset is packed ONCE into 4,096-token rows (prepare_sft_data.py --pack),
#     every optimizer step takes a fixed 20 rows (B 4 x 5 passes); chats per step vary.
#   B (fixed chats per step, per-step packing, the open-instruct / verl / Megatron-RL / NeMo-RL style): every step takes
#     a fixed 128 chats, packs THEM into 4,096-token rows (Best-Fit Decreasing inside the step), so rows (and passes at
#     B = 4) vary per step.
# Input: train/seq_lengths.bin (int32, every chat's length, in packed-row order) + train/seq_offsets.bin (int64 [rows+1])
#        + train/valid_counts.bin (uint32 [rows], trained tokens per row), copied from the box on 2026-10-23.
# usage: python3 simulate_packing.py <dir with seq_lengths.bin, seq_offsets.bin, valid_counts.bin>
import array, bisect, math, random, statistics, sys

d = sys.argv[1]
def load(name, code):
    a = array.array(code); 
    with open(f"{d}/{name}", "rb") as f: a.frombytes(f.read())
    return a
L = load("seq_lengths.bin", "i"); O = load("seq_offsets.bin", "q"); V = load("valid_counts.bin", "I")
T, ROWS_PER_STEP, CHATS_PER_STEP, B = 4096, 20, 128, 4
n_rows, n_chats, total = len(O) - 1, len(L), sum(L)
assert O[-1] == n_chats and len(V) == n_rows

def q(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]
def desc(xs, unit=""):
    return (f"mean {statistics.mean(xs):,.1f}{unit}  sd {statistics.pstdev(xs):,.1f}  min {min(xs):,}  p5 {q(xs, .05):,}  "
            f"median {q(xs, .5):,}  p95 {q(xs, .95):,}  max {max(xs):,}")

print(f"data: {n_chats:,} chats, mean length {total / n_chats:.1f} tokens, median {q(L, .5)}, p95 {q(L, .95)}, max {max(L)}")
print(f"      {total:,} real tokens; lower bound on rows = ceil(total / {T}) = {math.ceil(total / T):,}\n")

# ---------------- A: our offline packing (the file as trained), 20 rows per step in file order
row_used = [sum(L[O[r]:O[r + 1]]) for r in range(n_rows)]
row_chats = [O[r + 1] - O[r] for r in range(n_rows)]
stepsA = n_rows // ROWS_PER_STEP
chA = [sum(row_chats[s * 20:(s + 1) * 20]) for s in range(stepsA)]
tokA = [sum(row_used[s * 20:(s + 1) * 20]) for s in range(stepsA)]
trA = [sum(V[s * 20:(s + 1) * 20]) for s in range(stepsA)]
print("A  ours: offline packing of the whole dataset, fixed 20 rows per step")
print(f"   rows {n_rows:,} (lower bound + {n_rows - math.ceil(total / T)}), fill {100 * total / (n_rows * T):.3f}% of all row positions")
print(f"   steps per epoch {n_rows / ROWS_PER_STEP:,.1f}; passes per step: always {ROWS_PER_STEP // B} (B {B})")
print(f"   chats per step       : {desc(chA)}")
print(f"   real tokens per step : {desc(tokA)}")
print(f"   trained tokens / step: {desc(trA)}\n")

# ---------------- B: fixed 128 chats per step, packed per step with Best-Fit Decreasing
def bfd(lengths, cap):
    free = []                                  # sorted list of (free space) for open bins
    bins = 0
    for x in sorted(lengths, reverse=True):
        i = bisect.bisect_left(free, x)        # tightest bin that still fits
        if i < len(free):
            f = free.pop(i) - x
        else:
            bins += 1; f = cap - x
        if f > 0: bisect.insort(free, f)
    return bins

for seed in (1, 2):
    rnd = random.Random(seed); idx = list(range(n_chats)); rnd.shuffle(idx)
    stepsB = n_chats // CHATS_PER_STEP
    rowsB, tokB, passB = [], [], []
    for s in range(stepsB):
        ls = [L[i] for i in idx[s * CHATS_PER_STEP:(s + 1) * CHATS_PER_STEP]]
        r = bfd(ls, T); rowsB.append(r); tokB.append(sum(ls)); passB.append(math.ceil(r / B))
    used_rows = sum(rowsB); slots = sum(4 * p for p in passB)
    print(f"B  fixed {CHATS_PER_STEP} chats per step, per-step Best-Fit Decreasing (shuffle seed {seed})")
    print(f"   steps per epoch {n_chats / CHATS_PER_STEP:,.1f}; rows per epoch {used_rows:,} (+{100 * (used_rows / n_rows - 1):.2f}% vs A), "
          f"fill {100 * total / (used_rows * T):.3f}%")
    print(f"   rows per step        : {desc(rowsB)}")
    print(f"   passes per step (B 4): {desc(passB)}  -> row slots if every pass is a full B 4: {slots:,} "
          f"(+{100 * (slots / n_rows - 1):.2f}% vs A)")
    print(f"   real tokens per step : {desc(tokB)}")
    hist = {}
    for p in passB: hist[p] = hist.get(p, 0) + 1
    print("   passes histogram     : " + ", ".join(f"{k} passes: {v}" for k, v in sorted(hist.items())) + "\n")

# ---------------- B': per-step packing of a fixed number of chats, with NO packing (padded), for scale
padded_rows = n_chats   # one chat per 4,096 row
print(f"no packing (one chat per row, padded): {padded_rows:,} rows = {padded_rows / n_rows:.2f}x the rows of A")
