# Port SASS gate (2026-10-23), library flags (--use_fast_math -O3, sm_90a, nvcc 13.0):
#  (1) every kernel of the stage-B reference that the port keeps (all forwards, backward HAS_POS=false) must be IDENTICAL;
#  (2) the port's new kernels must be IDENTICAL to the kernels measured on 10-22 (ropt::bwd<..,true>, ropt2::reduce3<128>).
# usage: python3 port_sass_compare.py ref.sass new.sass inst.sass
import re, sys, subprocess
def funcs(path):
    out = {}; cur = None
    for line in open(path):
        m = re.match(r"\s*Function : (\S+)", line)
        if m: cur = m.group(1); out[cur] = []; continue
        if cur is None: continue
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;", line)
        if m: out[cur].append(m.group(1))
    return out
def dm(n): return subprocess.run(["c++filt", n], capture_output=True, text=True).stdout.strip().split("(")[0].replace("OwnTensor::cuda::", "").removeprefix("void ")
ref = {dm(k): v for k, v in funcs(sys.argv[1]).items()}
new = {dm(k): v for k, v in funcs(sys.argv[2]).items()}
ins = {dm(k): v for k, v in funcs(sys.argv[3]).items()}
bad = 0; n = 0
print("(1) kept kernels, stage-B reference vs port:")
for name in sorted(ref):
    if "bwd_bf16_tm" in name and name.endswith("128, true>"): print(f"   dropped (positions at hd 128 now use bwd_pf)   {name}"); continue
    if name.startswith("reduce_dgamma_bf16_tm"): print(f"   replaced by reduce_dgamma_qk<128>    {name}"); continue
    n += 1; eq = new.get(name) == ref[name]; bad += not eq
    print(f"   {'IDENTICAL' if eq else 'DIFFERENT'} {len(ref[name]):5d} vs {len(new.get(name, [])):5d} instr  {name}")
print("(2) new kernels, port vs the kernels measured on 10-22:")
pairs = [(k, k.replace("fused_qknorm_rope_bwd_pf_bf16_tm", "ropt::bwd")[:-1] + ", true>") for k in new if k.startswith("fused_qknorm_rope_bwd_pf")]
pairs += [("reduce_dgamma_qk_bf16_tm<128>", "ropt2::reduce3<128>")]
for a, b in sorted(pairs):
    n += 1; eq = new.get(a) == ins.get(b) and a in new and b in ins; bad += not eq
    print(f"   {'IDENTICAL' if eq else 'DIFFERENT'} {len(new.get(a, [])):5d} vs {len(ins.get(b, [])):5d} instr  {a}  vs  {b}")
print(f"RESULT: {n - bad}/{n} identical" + ("" if bad == 0 else "  <-- NOT PASSED"))
sys.exit(1 if bad else 0)
