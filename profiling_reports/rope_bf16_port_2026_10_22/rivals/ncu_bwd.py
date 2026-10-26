# One BACKWARD call of one implementation inside cudaProfilerStart/Stop, for ncu --profile-from-start off (2026-10-23).
# Step 3 of the RoPE work: torch.compile's backward beat ours in rival_bench.py (18x6 B1: 52.5 vs 91.9 us; 32x8 B4: 243.7
# vs 313.6 us), so both are profiled side by side on identical inputs. Reuses rival_bench.py's Case / Ours / Autograd, so
# the inputs and the calls are exactly those of the benchmark.
# usage: ncu --profile-from-start off --set full -o <rep> python ncu_bwd.py --impl ours|compile --B 4 --heads 18x6 [--packed]
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rival_bench as rb

ap = argparse.ArgumentParser()
ap.add_argument("--impl", choices=["ours", "compile"], required=True)
ap.add_argument("--B", type=int, default=4); ap.add_argument("--heads", default="18x6"); ap.add_argument("--T", type=int, default=4096)
ap.add_argument("--packed", action="store_true")
ap.add_argument("--lib", default="./libropeours_port.so")
a = ap.parse_args()
Hq, Hkv = map(int, a.heads.split("x"))
c = rb.Case(a.B, a.T, Hq, Hkv, a.packed, "cuda")
if a.impl == "ours":
    impl = rb.Ours(a.lib)
else:
    compiled = torch.compile(rb.hf_qk, dynamic=False)
    impl = rb.Autograd("compile", lambda c, q, k, wq, wk: compiled(q, k, wq, wk, c.cos_hf, c.sin_hf), rb.BF16, True)
fwd, bwd, _ = impl.prepare(c)
for _ in range(5):                      # warm-up (and compilation for torch.compile)
    fwd(); bwd()
fwd()                                   # the forward whose backward is profiled
torch.cuda.synchronize()
torch.cuda.profiler.start()
bwd()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print(f"profiled one backward: {a.impl} B {a.B} heads {a.heads} {'packed' if a.packed else 'pretraining'}")
