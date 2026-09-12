#!/usr/bin/env bash
# §H228  DOES `watch -n 0.1 nvidia-smi` ACTUALLY PERTURB OUR POWER MEASUREMENTS?
#
# WHY THIS EXISTS. I claimed the user's `watch -n 0.1 nvidia-smi` corrupted power readings.
# The written record does not support that. §H224b attributes the stuck 84.8 W readings to
# TWO failures, both mine: a leaked `nvidia-smi -lms 100` sampler that ran 7,545 s, and my own
# status polling during a live measurement. The user's watch is named nowhere. So the claim
# was asserted, not measured. This measures it.
#
# DESIGN -- it never touches the user's terminal. Instead of stopping their poller (which I
# cannot restart for them), it ADDS identical pollers and looks for a dose response:
#   arm LOW  = the pollers already present (their watch)
#   arm HIGH = the same, plus 3 more identical `watch -n 0.1 nvidia-smi` started by this script
# If 1 -> 4 pollers moves nothing, the mechanism does not exist at this dose and 0 -> 1 cannot
# matter either. If it moves, the dose response tells us how much each poller costs.
# A null control (LOW vs LOW) runs afterwards so a "significant" result cannot come from drift.
set -u
cd ~/gautam_k/blublas
export PATH=/usr/local/cuda/bin:$PATH
GPUIDX=${GPU:-1}; export CUDA_VISIBLE_DEVICES=$GPUIDX
SIG=${SIG:-"tn|3072|16384|8192|1|0|0|0|1|0|0|0|0"}
ENTRY=${ENTRY:-blublasBgemmSM90_bf16_tn_128x256x64_template}
LOCK=${LOCK:-1200}; ROUNDS=${ROUNDS:-8}; TARGET_S=${TARGET_S:-6}; ASSUME_TF=${ASSUME_TF:-600}
BIN=./build/h224_sweep
OUT=h228_out; mkdir -p $OUT
IT=$(python3 -c "
s='$SIG'.split('|'); M,N,K=int(s[1]),int(s[2]),int(s[3])
print(max(200,int($TARGET_S*$ASSUME_TF*1e12/(2.0*M*N*K))))")

EXTRA_PIDS=""
stop_extra(){ for p in $EXTRA_PIDS; do kill -KILL $p 2>/dev/null; done; EXTRA_PIDS=""; }
kill_tele(){ [ -n "${TELE:-}" ] && [ "${TELE:-0}" -gt 0 ] 2>/dev/null && kill "$TELE" 2>/dev/null; TELE=""; }
restore(){ nvidia-smi -i $GPUIDX -rgc >/dev/null 2>&1; }
cleanup(){ kill_tele; stop_extra; restore; }
trap cleanup EXIT INT TERM HUP

BASE_WATCH=$(pgrep -cf '[w]atch.*nvidia-smi')
echo "=== H228 watch-poller dose test   $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
echo "    GPU $GPUIDX   $SIG   iters=$IT (~${TARGET_S}s)   lock=${LOCK}MHz   rounds=$ROUNDS"
echo "    pollers already present (NOT touched): $BASE_WATCH"

start_extra(){   # add 3 more identical pollers
  for i in 1 2 3; do
    setsid nohup watch -n 0.1 nvidia-smi > /dev/null 2>&1 < /dev/null &
    EXTRA_PIDS="$EXTRA_PIDS $!"
  done
  sleep 2
}

measure(){   # echoes "power tflops belowlock%"
  nvidia-smi -i $GPUIDX --query-gpu=power.draw,clocks.sm,utilization.gpu \
    --format=csv,noheader,nounits -lms 100 > /tmp/h228.csv & TELE=$!
  local tf; tf=$($BIN bench "$ENTRY" "$SIG" $IT 2>/dev/null | awk '{print $1}')
  local T=$TELE; kill_tele; wait $T 2>/dev/null
  python3 -c "
import statistics
rows=[l.split(',') for l in open('/tmp/h228.csv') if l.strip()]
b=[(float(x[0]),float(x[1])) for x in rows if len(x)>=3 and float(x[2])>=95]
p=statistics.median([y[0] for y in b]) if b else 0
lo=100.0*sum(1 for y in b if y[1]<$LOCK-5)/len(b) if b else 100.0
print('%.1f %s %.1f'%(p,'${tf:-0}',lo))"
}

nvidia-smi -i $GPUIDX -lgc $LOCK >/dev/null 2>&1

run_pair(){   # $1 label, $2 = 1 if HIGH arm really adds pollers
  local tag=$1 hi=$2
  : > $OUT/$tag.txt
  for r in $(seq 1 $ROUNDS); do
    # alternate order every round so order cannot be confounded with arm
    if [ $((r % 2)) -eq 1 ]; then order="LOW HIGH"; else order="HIGH LOW"; fi
    for arm in $order; do
      if [ "$arm" = "HIGH" ] && [ "$hi" = "1" ]; then start_extra; fi
      n=$(pgrep -cf '[w]atch.*nvidia-smi')
      read p t lo <<< "$(measure)"
      echo "$r|$arm|$n|$p|$t|$lo" >> $OUT/$tag.txt
      stop_extra; sleep 1
    done
    printf "  %-5s round %2d done\n" "$tag" "$r"
  done
  python3 - "$OUT/$tag.txt" "$tag" <<'PY'
import sys,statistics
rows=[l.strip().split('|') for l in open(sys.argv[1]) if l.strip()]
d={}
for r,arm,n,p,t,lo in rows: d.setdefault((int(r),arm),[]).append((float(p),float(t),int(n),float(lo)))
rounds=sorted({r for r,_ in d})
dp=[];dt=[];npoll={}
for r in rounds:
    if (r,'LOW') in d and (r,'HIGH') in d:
        lo=d[(r,'LOW')][0]; hi=d[(r,'HIGH')][0]
        dp.append(100*(hi[0]-lo[0])/lo[0]); dt.append(100*(hi[1]-lo[1])/lo[1])
        npoll[r]=(lo[2],hi[2])
def stat(v,nm):
    if len(v)<2: print("    %s: too few"%nm); return
    m=statistics.mean(v); sd=statistics.stdev(v); se=sd/len(v)**.5
    print("    %-11s mean %+7.3f%%  sd %.3f  se %.3f  t = %+6.2f  n=%d"%(nm,m,sd,se,(m/se if se else 0),len(v)))
print("  --- %s : pollers LOW=%s HIGH=%s"%(sys.argv[2],
      npoll[rounds[0]][0] if rounds else '?', npoll[rounds[0]][1] if rounds else '?'))
print("    max below-lock: %.1f%%"%max(x[3] for v in d.values() for x in v))
stat(dp,"power");stat(dt,"throughput")
print("    per-round power deltas: "+" ".join("%+.2f"%x for x in dp))
print("    VERDICT: %s"%("SIGNIFICANT (|t|>3) -- pollers DO perturb" if abs(statistics.mean(dp)/(statistics.stdev(dp)/len(dp)**.5))>3 else "NOT significant (|t|<=3) -- no detectable effect"))
PY
}

echo; echo "--- REAL: 1 poller vs 4 pollers ---"
run_pair real 1
echo; echo "--- NULL CONTROL: 1 poller vs 1 poller (must come out non-significant) ---"
run_pair null 0
restore; echo "[clocks free]  $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
