# 2026-10-19: per-step clock stamps of FA3's dense forward (the measurement build of fa3_tl_build.sh, work doc 45.51), in the
# regime of our own trace (trace_1017.sh: forward + backward back to back, 1,000 iterations, the forward of iterations 900 and
# 950 written out). Inputs and calls = the scoreboard's (rivals_free_1016.dense_contender: B 8, 4096 tokens, 18 / 6 heads,
# head dim 128, causal, bf16). The files use the layout of our V62_TRACE dumps (16 int64 header, an empty per-CTA block, no
# records, then the timeline), so deep/timeline_read.py reads FA3's and ours the same way.
# (level 2: stamps s0 / s1, s2 = s1; level 3: + s2, see fa3_tl_build.sh)
# usage (from ~/gautam_K/step6, for bench_step6_libs): python fa3_tl_run.py <traced hopper folder> <output folder>
import ctypes, os, struct, sys, time
TL_HOPPER, OUT = os.path.abspath(sys.argv[1]), sys.argv[2]
sys.path.insert(0, TL_HOPPER)                                     # the traced build first; benv's flash_attn_3 is not used
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))   # rivals_free_1016.py (this folder)
import torch
import flash_attn_3._C as C
import flash_attn_interface
assert C.__file__.startswith(TL_HOPPER), f"wrong flash_attn_3 loaded: {C.__file__}"
assert flash_attn_interface.__file__.startswith(TL_HOPPER), f"wrong interface loaded: {flash_attn_interface.__file__}"
import rivals_free_1016 as R
lib = ctypes.CDLL(C.__file__)
lib.fa3_tl_read.argtypes = [ctypes.c_void_p, ctypes.c_size_t]; lib.fa3_tl_read.restype = ctypes.c_int
LEVEL = lib.fa3_tl_level()
NS, TLS, TLW = 132, 64, 8
NW = lib.fa3_tl_words(); assert NW == NS * 2 * TLS * TLW, f"trace buffer {NW} words, expected {NS * 2 * TLS * TLW}"
S = R.S
print("torch", torch.__version__, "| gpu", torch.cuda.get_device_name(0), "| FA3 traced build", C.__file__, "| level", LEVEL, flush=True)
f, b = R.dense_contender("fa3")
os.makedirs(OUT, exist_ok=True)
for _ in range(20):                                           # warm-up, as rivals_free_1016
    f(); b()
torch.cuda.synchronize()
t0 = time.time(); last = t0
for it in range(1, 1001):                                     # back to back, no synchronisation (except at the two dumps)
    f(); b()
    if it in (900, 950):
        torch.cuda.synchronize(); now = time.time()
        buf = (ctypes.c_uint32 * NW)()
        rc = lib.fa3_tl_read(buf, NW * 4)
        hdr = struct.pack("<16q", 0x46413354, 0, S.B, S.HQ, S.HKV, S.T, 0, NS, 0, 0, 0, 0, 0, 0, 0, 0)
        path = os.path.join(OUT, f"trace_dense_{it}.bin")
        with open(path, "wb") as fo:
            fo.write(hdr + bytes(NS * 2 * 4 * 8) + bytes(buf))
        words = list(buf); filled = sum(1 for i in range(0, NW, TLW) if words[i + 5] != 0)
        print(f"TRACE iteration {it} -> {path} | copy rc={rc} | steps with stamps {filled} | taken at "
              f"{time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(now))}.{int(now * 1000) % 1000:03d} | "
              f"{1000 * (now - last) / (it - (900 if it == 950 else 0)):.4f} ms per iteration since the last mark", flush=True)
        last = now
torch.cuda.synchronize()
dt = time.time() - t0
print(f"FA3TL done: 1000 forward + backward in {dt:.2f} s, {1000 * dt / 1000:.4f} ms per iteration", flush=True)
