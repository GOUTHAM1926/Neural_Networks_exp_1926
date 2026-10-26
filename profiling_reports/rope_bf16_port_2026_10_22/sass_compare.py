# Stage B SASS check (2026-10-22): every kernel of Kruthik's original FusedRMSRoPE_bf16.cu vs the HAS_POS=false
# instantiation of the stage-B file. usage: python3 sass_compare.py orig.sass new.sass  (cuobjdump -sass of each cubin)
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
def dm(n): return subprocess.run(["c++filt", n], capture_output=True, text=True).stdout.strip()
o = {dm(k): v for k, v in funcs(sys.argv[1]).items()}; n = {dm(k): v for k, v in funcs(sys.argv[2]).items()}
same = 0
for name, ins in sorted(o.items()):
    tmpl = name.split("(")[0]
    want = tmpl[:-1] + ", false>" if tmpl.endswith(">") else tmpl
    m = [k for k in n if k.split("(")[0] == want]; assert len(m) == 1, (name, m)
    eq = n[m[0]] == ins; same += eq
    print(("IDENTICAL " if eq else "DIFFERENT ") + f"{len(ins):5d} vs {len(n[m[0]]):5d} instr  {tmpl.replace('OwnTensor::cuda::','')}")
print(f"{same}/{len(o)} kernels SASS-identical (HAS_POS=false vs Kruthik's original)")
for k, v in sorted(n.items()):
    t = k.split("(")[0]
    if t.endswith("true>"):
        base = [o[x] for x in o if x.split("(")[0] == t[:-len(", true>")] + ">"][0]
        print(f"  HAS_POS=true  {len(v):5d} instr (original {len(base)}, +{len(v)-len(base)})  {t.replace('OwnTensor::cuda::','')}")
