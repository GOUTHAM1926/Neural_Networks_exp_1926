# Row of 4096: chat 1 = tokens 0-999, chat 2 = tokens 1000-3999, pad tail 4000-4095. Forward blocks 64 queries x 128 keys,
# rules of GQA_fwd_sm90.cu (segstart: key loop of a 128-row query pair starts at the key block holding the chat start
# of the pair's first row; seqlens: grid restarts at every chat).
T = 4096; segs = [1000, 3000, 96]
seg = []; t = 0
for l in segs: seg += [t] * l; t += l
def cdiv(a, b): return -(-a // b)
# seqlens
for name, l in zip(("chat 1", "chat 2", "pad"), segs):
    print(f"seqlens {name}: {sum((64*qt+63)//128 + 1 for qt in range(cdiv(l, 64)))} blocks")
# segstart: per query block, which key blocks, and how many keys of other chats
tot = 0; extra_blocks = 0
for qt in range(T // 64):
    n0 = seg[128 * (qt // 2)] // 128
    last = (64 * qt + 63) // 128
    nb = last + 1 - n0; tot += nb
    rows = range(64 * qt, 64 * qt + 64)
    # a key block is "extra" for this query block when NONE of its rows may see any key in it
    for kb in range(n0, last + 1):
        keys = range(128 * kb, 128 * kb + 128)
        if not any(seg[r] <= k <= r for r in rows for k in (keys[0], keys[-1]) ) and not any(seg[r] <= k <= r for r in rows[::8] for k in keys[::8]):
            extra_blocks += 1
print(f"segstart whole row: {tot} blocks")
print(f"seqlens  whole row: {sum(sum((64*qt+63)//128 + 1 for qt in range(cdiv(l, 64))) for l in segs)} blocks")
# the query block in the MIDDLE of chat 2, rows 2048-2111
qt = 2048 // 64
n0 = seg[128 * (qt // 2)] // 128; last = (64 * qt + 63) // 128
print(f"\nmiddle query block of chat 2 (rows 2048-2111):")
print(f"  segstart key blocks: {n0}..{last} = keys {128*n0}-{128*last+127} -> {last+1-n0} blocks; first one holds keys {128*n0}-{128*n0+127}: "
      f"{sum(1 for k in range(128*n0, 128*n0+128) if seg[k] != 1000)} of chat 1 (masked), {sum(1 for k in range(128*n0, 128*n0+128) if seg[k] == 1000)} of chat 2")
r0 = 2048 - 1000
print(f"  seqlens  key blocks: chat-2 keys {1000}-{1000 + 128*((r0+63)//128) + 127} -> {(r0 + 63)//128 + 1} blocks, all chat 2")
