# One 4096-token row filled with chats of length L (rest = pad tail). Forward blocks are 64 queries x 128 keys,
# following the kernels' rules (GQA_fwd_sm90.cu): seqlens restarts the block grid at every chat, segstart keeps it on the row.
T = 4096
def cdiv(a, b): return -(-a // b)
print(f"{'chat length':>11} {'chats':>6} | {'seqlens: blocks':>15} {'real share':>10} | {'segstart: blocks':>16} {'real share':>10}")
for L in (20, 64, 100, 300, 1000, 2000):
    n = T // L; segs = [L] * n + ([T - n * L] if T - n * L else [])
    real = n * L * (L + 1) // 2
    a = sum(sum((64 * qt + 63) // 128 + 1 for qt in range(cdiv(l, 64))) for l in segs)      # seqlens blocks
    seg = []; t = 0
    for l in segs: seg += [t] * l; t += l
    b = sum((64 * qt + 63) // 128 + 1 - seg[128 * (qt // 2)] // 128 for qt in range(T // 64))   # segstart blocks
    print(f"{L:>11} {n:>6} | {a:>15} {100 * real / (a * 8192):>9.1f}% | {b:>16} {100 * real / (b * 8192):>9.1f}%")
