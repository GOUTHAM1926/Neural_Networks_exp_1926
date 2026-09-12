#!/usr/bin/env bash
# §H220 REUSABLE POWER A/B between two libraries. Encodes every correction from 2026-09-11.
#
#   A=<baseline.so> B=<candidate.so> ARG=<template-arg-index> [LOCK=1200] [ROUNDS=3] bash h220_ab.sh
#
# FOUR RULES BUILT IN, each of which cost a wrong result today:
#
# 1. INDEPENDENT OPERANDS (H211_FILL=3). The old fill copied one host buffer into both A and B
#    with a generator periodic mod 2048. cuBLAS's 256x128 stride landed in phase with that
#    period, its operands barely toggled, and a shape that is really -0.8% measured +41.3%.
#    Benchmark input is part of the experiment.
# 2. LOCK BELOW THE CAP, and ASSERT IT HELD. At 1400 MHz with realistic operands both arms sit
#    at ~690-700 W, i.e. AT the 700 W cap, which clips and compresses the gap. 1200 MHz is
#    clean. Separately, an -lgc that silently did not take once produced 843 TFLOPS against a
#    735 TFLOPS ceiling and faked a 32% effect -- so the achieved clock and a below-lock
#    percentage are printed for every single measurement.
# 3. VERIFY THE TWO LIBRARIES DIFFER IN THE INTENDED PARAMETER AND NOTHING ELSE, by reading the
#    kernel's template arguments out of ncu. A build that did not take looks exactly like a null.
# 4. CORRECTNESS BEFORE TIMING, bit-exact. A faster wrong kernel is not a result.
set -u
cd ~/gautam_k/blublas
export PATH=/usr/local/cuda/bin:$PATH
GPUIDX=${GPU:-0}; export CUDA_VISIBLE_DEVICES=$GPUIDX
A=${A:?set A=<baseline .so>}; Bso=${B:?set B=<candidate .so>}
ARG=${ARG:-21}; LOCK=${LOCK:-1200}; ROUNDS=${ROUNDS:-3}; TARGET_S=${TARGET_S:-2.0}
CASES=${CASES:-tools/h212_cases.txt}
BIN=./build/h211_replay
OUT=h220_out; mkdir -p $OUT
for f in "$A" "$Bso" "$BIN" "$CASES"; do [ -e "$f" ] || { echo "!! missing $f"; exit 1; }; done
use(){ cp -f "$1" build/libmyblas.so; }
restore(){ nvidia-smi -i $GPUIDX -rgc >/dev/null 2>&1; use "$A"; echo "  [clocks free, baseline restored]"; }
trap 'kill ${TELE:-0} 2>/dev/null; restore' EXIT INT TERM

echo "=== H220 A/B   A=$(basename $A)  B=$(basename $Bso)  arg[$ARG]  lock=${LOCK}  $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
echo "--- 0. do the libraries differ in arg[$ARG] and nothing else? ---"
for arm in A B; do
  [ $arm = A ] && use "$A" || use "$Bso"
  /usr/local/cuda/bin/ncu --metrics smsp__inst_executed.sum --launch-skip 5 --launch-count 1 \
    --clock-control base --csv --target-processes all \
    $BIN ours "nn|8192|3072|8192|1|0|0|0|1|0|0|0|25165824" 12 2>/dev/null > /tmp/ab_$arm.csv
  python3 - "$arm" "$ARG" <<'PY'
import sys,csv,io,re
arm,idx=sys.argv[1],int(sys.argv[2])
raw=open("/tmp/ab_%s.csv"%arm,errors="ignore").read().splitlines()
h=next((i for i,l in enumerate(raw) if l.startswith('"ID"')),None)
if h is None: print("    arm %s: NO NCU DATA"%arm); raise SystemExit
nm=next(csv.DictReader(io.StringIO("\n".join(raw[h:]))))["Kernel Name"]
a=[x.strip() for x in re.search(r"Sm90WarpspecConfig<([^>]*)>",nm).group(1).split(",")]
open("/tmp/args_%s.txt"%arm,"w").write(",".join(a))
print("    arm %s: arg[%d] = %s"%(arm,idx,a[idx] if idx<len(a) else "?"))
PY
done
python3 - "$ARG" <<'PY'
import sys
idx=int(sys.argv[1])
a=open("/tmp/args_A.txt").read().split(","); b=open("/tmp/args_B.txt").read().split(",")
d=[i for i in range(min(len(a),len(b))) if a[i]!=b[i]]
print("    template args differing: %s"%(d if d else "NONE"))
if d==[idx]: print("    OK -- exactly the intended parameter")
elif not d: print("    !! IDENTICAL -- the build did not take; a null here would be meaningless")
else: print("    !! more than the intended parameter changed: %s"%d)
PY

echo "--- 1. correctness of arm B, every shape, must be bit-exact (independent operands) ---"
use "$Bso"; bad=0
while IFS= read -r line; do
  case "$line" in \#*|"") continue;; esac
  sig=$(echo "$line"|sed 's/[[:space:]]*#.*//'); [ -z "$sig" ] && continue
  r=$(H211_FILL=3 $BIN check "$sig" 2>&1|tail -1)
  case "$r" in PASS*err=0.000e+00*) :;; *) echo "    !! $sig -> $r"; bad=1;; esac
done < "$CASES"
[ $bad = 0 ] && echo "    all bit-exact" || { echo "  CORRECTNESS FAILED -- stopping"; exit 1; }

echo "--- 2. power, locked ${LOCK} MHz, fill=3, ${ROUNDS}-round ABBA ---"
RAW=$OUT/raw.txt; : > $RAW
nvidia-smi -i $GPUIDX -lgc $LOCK >/dev/null 2>&1
echo "  site             shape                A power   B power    dPower  belowlock  clk"
while IFS= read -r line; do
  case "$line" in \#*|"") continue;; esac
  sig=$(echo "$line"|sed 's/[[:space:]]*#.*//'); tag=$(echo "$line"|sed -n 's/.*#[[:space:]]*//p')
  [ -z "$sig" ] && continue
  M=$(echo "$sig"|cut -d'|' -f2); N=$(echo "$sig"|cut -d'|' -f3); K=$(echo "$sig"|cut -d'|' -f4)
  IT=$(python3 -c "f=2*$M*$N*$K; print(max(20,min(200000,int($TARGET_S*640e12/f))))")
  for r in $(seq 1 $ROUNDS); do
    for arm in A B B A; do
      [ $arm = A ] && use "$A" || use "$Bso"
      nvidia-smi -i $GPUIDX --query-gpu=power.draw,clocks.sm,utilization.gpu \
        --format=csv,noheader,nounits -lms 100 > /tmp/t.csv & TELE=$!
      TF=$(H211_FILL=3 $BIN ours "$sig" $IT 2>/dev/null | awk '{print $1}')
      kill $TELE 2>/dev/null; wait $TELE 2>/dev/null
      python3 -c "
import statistics,sys
b=[(float(x[0]),float(x[1])) for x in (l.split(',') for l in open('/tmp/t.csv') if l.strip()) if len(x)>=3 and float(x[2])>=95]
if b:
    lo=100.0*sum(1 for y in b if y[1]<$LOCK-20)/len(b)
    print('$sig|$arm|%.2f|%.0f|%.1f|$TF'%(statistics.median([y[0] for y in b]),statistics.median([y[1] for y in b]),lo))
" >> $RAW
    done
  done
  python3 - "$tag" "$(echo $sig|cut -d'|' -f1) ${M}x${N}x${K}" "$sig" $RAW <<'PY'
import sys,statistics
tag,shp,sig,raw=sys.argv[1:5]
rows=[l.strip().split("|") for l in open(raw)]
g={}
for r in rows:
    if "|".join(r[:-5])==sig: g.setdefault(r[-5],[]).append((float(r[-4]),float(r[-3]),float(r[-2]),float(r[-1])))
A=[x[0] for x in g.get("A",[])]; B=[x[0] for x in g.get("B",[])]
cl=[x[1] for x in g.get("A",[])]; lo=[x[2] for x in g.get("A",[])]
if A and B:
    a,b=statistics.median(A),statistics.median(B)
    print("  %-16s %-20s %7.1fW %8.1fW %+8.2f%%  %4.1f%%  %5.0f"%(tag,shp,a,b,(b/a-1)*100,max(lo),statistics.median(cl)))
PY
done < "$CASES"
restore
echo
python3 - $RAW <<'PY'
import sys,statistics,collections
rows=[l.strip().split("|") for l in open(sys.argv[1])]
g=collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    if len(r)<18: continue
    g["|".join(r[:-5])][r[-5]].append((float(r[-4]),float(r[-1])))
out=[]
for sig,d in g.items():
    if "A" in d and "B" in d:
        a,b=statistics.median([x[0] for x in d["A"]]),statistics.median([x[0] for x in d["B"]])
        ta,tb=statistics.median([x[1] for x in d["A"]]),statistics.median([x[1] for x in d["B"]])
        f=sig.split("|"); out.append((sig,a,b,(b/a-1)*100,2.0*int(f[1])*int(f[2])*int(f[3]),(tb/ta-1)*100 if ta else 0))
if out:
    ws=sum(o[4] for o in out)
    print("  FLOP-weighted:  power %+.2f%%   throughput %+.2f%%"%(sum(o[3]*o[4] for o in out)/ws, sum(o[5]*o[4] for o in out)/ws))
    print("  shapes where B is lower: %d/%d"%(sum(1 for o in out if o[3]<0),len(out)))
PY
