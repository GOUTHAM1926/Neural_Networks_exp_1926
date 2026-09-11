#!/usr/bin/env bash
# §H225 SWEEP EVERY CALLABLE VARIANT OF ONE TILE ON ITS PRODUCTION SHAPE.
#
# WHY. Knob-by-knob A/B has now returned six nulls and one -0.43% keeper, and the cluster
# sweep refuted the last structural lead. Rather than keep guessing which knob to try next,
# enumerate the ENTIRE registered variant space for the tile and let the measurement rank it.
# The dlsym harness makes this cheap: the library already exports every config, so there is no
# rebuild and no dispatcher involvement.
#
# TWO PASSES, deliberately:
#   SCREEN  1 round each over all variants -- finds candidates, cannot settle small differences
#   CONFIRM 3+ rounds on the top few -- a partial ABBA settling from -2.85% to -0.51% earlier
#           today is exactly why a screen is never quoted as a result
#
# Correctness gates every variant. A missing symbol or a launch failure is reported, not
# silently skipped: a config that cannot run is a finding about the registry, not a null.
set -u
cd ~/gautam_k/blublas
export PATH=/usr/local/cuda/bin:$PATH
GPUIDX=${GPU:-0}; export CUDA_VISIBLE_DEVICES=$GPUIDX
SIG=${SIG:-"tn|3072|3072|8192|1|0|0|0|1|0|0|0|0"}
PREFIX=${PREFIX:-blublasBgemmSM90_bf16_tn_192x192x64}
LOCK=${LOCK:-1200}; IT=${IT:-7800}; TOPN=${TOPN:-6}; CONF=${CONF:-3}
VTMO=${VTMO:-60}   # seconds per variant; see the per-variant timeout note in measure()
BIN=./build/h224_sweep
OUT=h225_out; mkdir -p $OUT
cp -f /tmp/so_A_li_dh0.so build/libmyblas.so
restore(){ nvidia-smi -i $GPUIDX -rgc >/dev/null 2>&1; }
cleanup(){ kill ${TELE:-0} 2>/dev/null; pkill -f "[n]vidia-smi.*lms.*power.draw" 2>/dev/null; restore; }
trap cleanup EXIT INT TERM HUP

# §H224b a leaked nvidia-smi -lms sampler ran for 7,545 s through several measurements and
# made the power sensor return stuck values (19 busy samples all exactly 84.8 W under a
# 609 TFLOPS load). Refuse to start rather than silently measure noise.
STRAY=$(pgrep -f "[n]vidia-smi.*lms" | wc -l)
if [ "$STRAY" != "0" ]; then
  echo "!! $STRAY nvidia-smi sampler(s) already polling the GPU -- power readings would be"
  echo "   corrupted. Clear them first:  pgrep -f '[n]vidia-smi.*lms'"
  exit 1
fi
nm -D --defined-only build/libmyblas.so | grep -oE "${PREFIX}[A-Za-z0-9_]*" | sort -u > $OUT/vars.txt
N=$(wc -l < $OUT/vars.txt)
echo "=== H225 sweep  $PREFIX  on $SIG   variants=$N   $(TZ=Asia/Kolkata date +%H:%M:%S) IST"

measure () {   # $1 entry ; echoes "power tflops"
  nvidia-smi -i $GPUIDX --query-gpu=power.draw,clocks.sm,utilization.gpu \
    --format=csv,noheader,nounits -lms 100 > /tmp/h225.csv & TELE=$!
  local tf
  # §H224b PER-VARIANT TIMEOUT. tn_192x192x64_2x1_h_ea32s2stsm_rs8_dh_ubd_skew_slp ran 394 s
  # without completing and stalled the 95 variants behind it. The `slp` arm is MBAR_SLEEP_TICKS,
  # which §H39 already records as engaging but losing. A registry sweep must assume some
  # registered config does not terminate.
  if [ "$1" = "cublas" ]; then tf=$(timeout $VTMO $BIN cublas "$SIG" $IT 2>/dev/null | awk '{print $1}')
  else tf=$(timeout $VTMO $BIN bench "$1" "$SIG" $IT 2>/dev/null | awk '{print $1}'); fi
  [ -z "$tf" ] && tf=0
  kill $TELE 2>/dev/null; wait $TELE 2>/dev/null
  python3 -c "
import statistics
b=[(float(x[0]),float(x[1])) for x in (l.split(',') for l in open('/tmp/h225.csv') if l.strip()) if len(x)>=3 and float(x[2])>=95]
print('%.1f %s'%(statistics.median([y[0] for y in b]) if b else 0,'${tf:-0}'))"
}

nvidia-smi -i $GPUIDX -lgc $LOCK >/dev/null 2>&1
echo "--- SCREEN: correctness + 1 round each ---"
: > $OUT/screen.txt
i=0
while read -r v; do
  i=$((i+1))
  r=$(timeout $VTMO $BIN check "$v" "$SIG" 2>&1 | tail -1 | awk '{print $1}')
  [ -z "$r" ] && r=TIMEOUT
  if [ "$r" != "PASS" ]; then printf "  [%3d/%3d] %-52s %s\n" "$i" "$N" "$(echo $v|sed "s/${PREFIX}_*//")" "$r"; continue; fi
  read p t <<< "$(measure "$v")"
  echo "$v|$p|$t" >> $OUT/screen.txt
  printf "  [%3d/%3d] %-52s %7sW %8s TF\n" "$i" "$N" "$(echo $v|sed "s/${PREFIX}_*//")" "$p" "$t"
done < $OUT/vars.txt
read p t <<< "$(measure cublas)"; echo "cublas|$p|$t" >> $OUT/screen.txt
printf "  %-64s %7sW %8s TF\n" "cuBLAS reference" "$p" "$t"

echo
echo "--- top $TOPN by efficiency (TFLOPS/W), from the screen ---"
sort -t'|' -k3 -g $OUT/screen.txt | awk -F'|' '$2>100{printf "%.5f|%s|%s|%s\n",$3/$2,$1,$2,$3}' \
  | sort -t'|' -k1 -gr | head -$((TOPN+1)) | tee $OUT/top.txt \
  | awk -F'|' '{printf "  %8.4f TF/W  %7sW %8s TF  %s\n",$1,$3,$4,$2}'

echo
echo "--- CONFIRM: $CONF rounds on the top $TOPN plus the production baseline ---"
CAND=$(awk -F'|' '{print $2}' $OUT/top.txt | head -$TOPN)
CAND="$CAND ${PREFIX}_template cublas"
for v in $(echo "$CAND" | tr ' ' '\n' | awk '!seen[$0]++'); do
  PS=""; TS=""
  for r in $(seq 1 $CONF); do read p t <<< "$(measure "$v")"; PS="$PS $p"; TS="$TS $t"; done
  python3 - "$(echo $v|sed "s/${PREFIX}_*//")" "$PS" "$TS" <<'PY'
import sys,statistics
s=sys.argv[1]; P=[float(x) for x in sys.argv[2].split() if x]; T=[float(x) for x in sys.argv[3].split() if x]
if P and T:
    print("  %-46s %7.1fW %8.1f TF  %7.4f TF/W  (spread %.1fW)"%(
        s,statistics.median(P),statistics.median(T),statistics.median(T)/statistics.median(P),max(P)-min(P)))
PY
done
restore; echo "[clocks free]"
