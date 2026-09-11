#!/usr/bin/env bash
# §H226 12-ROUND PAIRED CROSSOVER with a NULL CONTROL.
#
# THE CLAIM UNDER TEST. §H225 screened 131 tn 192x192 variants and found
# `2x1_h_rs8_dh_ubd_dw_skew_lsw_pdl_wl` at +0.57% throughput and -1.2% power over the config
# the dispatcher actually picks. That is NOT yet a result: the same production config measured
# 517.8 W in one run and 523.6 W in the next -- 1.1% run-to-run drift -- and the candidate's own
# spread was 5.0 W. A 1.8% efficiency claim resting on ~1% noise needs settling properly.
#
# WHY 12 ROUNDS AND A CROSSOVER. A 6-round ABBA has previously returned t=4.63 on a KNOWN NULL
# (feedback_paired_null_instrument). Rounds alternate ABBA and BAAB so any monotone drift --
# warming, clock settling, cache state -- cancels rather than loading onto whichever arm runs
# first.
#
# WHY A NULL CONTROL. The same A is run against ITSELF for the same 12 rounds. If the
# instrument reports a significant difference there, it cannot be trusted on the real pair.
# This is the arm that "looks identical on failure" and is therefore the one that proves the
# test is not vacuous.
set -u
cd ~/gautam_k/blublas
export PATH=/usr/local/cuda/bin:$PATH
GPUIDX=${GPU:-0}; export CUDA_VISIBLE_DEVICES=$GPUIDX
SIG=${SIG:-"tn|3072|3072|8192|1|0|0|0|1|0|0|0|0"}
P=blublasBgemmSM90_bf16_tn_192x192x64
A=${A:-${P}_template}
CAND=${CAND:-${P}_2x1_h_rs8_dh_ubd_dw_skew_lsw_pdl_wl_template}
LOCK=${LOCK:-1200}; ROUNDS=${ROUNDS:-12}; IT=${IT:-7800}; VTMO=${VTMO:-60}
BIN=./build/h224_sweep; OUT=h226_out; mkdir -p $OUT
cp -f /tmp/so_A_li_dh0.so build/libmyblas.so
S=$(ps -eo cmd | grep -c "[n]vidia-smi.*lms" || true)
[ "$S" = "0" ] || { echo "!! $S stray sampler(s) polling the GPU -- refusing to start"; exit 1; }
cleanup(){ kill ${TELE:-0} 2>/dev/null; nvidia-smi -i $GPUIDX -rgc >/dev/null 2>&1; }
trap cleanup EXIT INT TERM HUP

one () {   # $1 entry -> "power tflops clk below"
  nvidia-smi -i $GPUIDX --query-gpu=power.draw,clocks.sm,utilization.gpu \
    --format=csv,noheader,nounits -lms 100 > /tmp/h226.csv & TELE=$!
  local tf; tf=$(timeout $VTMO $BIN bench "$1" "$SIG" $IT 2>/dev/null | awk '{print $1}')
  kill $TELE 2>/dev/null; wait $TELE 2>/dev/null
  python3 -c "
import statistics
b=[(float(x[0]),float(x[1])) for x in (l.split(',') for l in open('/tmp/h226.csv') if l.strip()) if len(x)>=3 and float(x[2])>=95]
print('%.2f %s %.0f %.1f'%(statistics.median([y[0] for y in b]),'${tf:-0}',statistics.median([y[1] for y in b]),100.0*sum(1 for y in b if y[1]<$LOCK-20)/len(b)) if b else '0 0 0 0')"
}

nvidia-smi -i $GPUIDX -lgc $LOCK >/dev/null 2>&1
run_pair () {   # $1 armA  $2 armB  $3 label
  local RAW=$OUT/$3.txt; : > $RAW
  echo "  --- $3 : $(echo $1|sed "s/${P}_*//")  vs  $(echo $2|sed "s/${P}_*//")"
  for r in $(seq 1 $ROUNDS); do
    # alternate ABBA / BAAB so drift cancels instead of loading onto one arm
    if [ $((r % 2)) -eq 1 ]; then ORD="A B B A"; else ORD="B A A B"; fi
    for a in $ORD; do
      [ "$a" = "A" ] && e=$1 || e=$2
      read p t c lo <<< "$(one $e)"
      echo "$r|$a|$p|$t|$c|$lo" >> $RAW
    done
    printf "    round %-2s done\n" "$r"
  done
  python3 - "$RAW" "$3" <<'PY'
import sys,statistics,collections
raw,lab=sys.argv[1],sys.argv[2]
R=collections.defaultdict(dict)
for l in open(raw):
    f=l.strip().split("|")
    if len(f)<6: continue
    R[int(f[0])].setdefault(f[1],[]).append((float(f[2]),float(f[3]),float(f[4]),float(f[5])))
dp=[];dt=[]
for r,d in sorted(R.items()):
    if "A" not in d or "B" not in d: continue
    pa=statistics.mean([x[0] for x in d["A"]]); pb=statistics.mean([x[0] for x in d["B"]])
    ta=statistics.mean([x[1] for x in d["A"]]); tb=statistics.mean([x[1] for x in d["B"]])
    dp.append((pb/pa-1)*100); dt.append((tb/ta-1)*100)
clk=[x[2] for d in R.values() for v in d.values() for x in v]
lo =[x[3] for d in R.values() for v in d.values() for x in v]
def stat(v,name):
    m=statistics.mean(v); s=statistics.stdev(v) if len(v)>1 else 0
    se=s/len(v)**0.5 if len(v)>1 else 0
    t=m/se if se else 0
    print("    %-12s mean %+7.3f%%   sd %5.3f   se %5.3f   t = %+6.2f   n=%d"%(name,m,s,se,t,len(v)))
    return m,t
print("    clock %.0f MHz, max below-lock %.1f%%"%(statistics.median(clk),max(lo)))
mp,tp=stat(dp,"power");  mt,tt=stat(dt,"throughput")
print("    per-round power deltas: %s"%" ".join("%+.2f"%x for x in dp))
print("    VERDICT: %s"%("significant (|t|>3)" if abs(tp)>3 or abs(tt)>3 else "NOT significant (|t|<=3)"))
PY
}
echo "=== H226 paired crossover, $ROUNDS rounds  $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
run_pair "$A" "$CAND" real
echo
echo "=== NULL CONTROL: the SAME config against itself -- must come out non-significant ==="
run_pair "$A" "$A" null
cleanup; echo "[clocks free]"
