#!/usr/bin/env bash
# §H224 CLUSTER SWEEP on the one shape where cuBLAS uses OUR EXACT TILE.
#
# tn 3072x3072x8192 (Q/O wgrad): cuBLAS runs nvjet_sm90_tst_192x192_64x3_1x2_h_bz_coopB_NTN --
# the same 192x192 tile, cluster 1x2, where ours is 1x1 -- and we measure +6.00% power there.
# Same tile, same shape, one structural difference. No other production shape offers that.
#
# 3072/192 = 16 tiles on BOTH axes, so EVERY cluster size here divides evenly and none of them
# pays the round-up penalty that makes cluster comparisons dishonest elsewhere.
#
# Correctness gates every variant before it is timed, and the achieved clock and below-lock
# percentage are printed per measurement.
set -u
cd ~/gautam_k/blublas
export PATH=/usr/local/cuda/bin:$PATH
GPUIDX=${GPU:-0}; export CUDA_VISIBLE_DEVICES=$GPUIDX
SIG=${SIG:-"tn|3072|3072|8192|1|0|0|0|1|0|0|0|0"}
LOCK=${LOCK:-1200}; ROUNDS=${ROUNDS:-3}; IT=${IT:-2200}
BIN=./build/h224_sweep
cp -f /tmp/so_A_li_dh0.so build/libmyblas.so
restore(){ nvidia-smi -i $GPUIDX -rgc >/dev/null 2>&1; }
trap 'kill ${TELE:-0} 2>/dev/null; restore' EXIT INT TERM

P=blublasBgemmSM90_bf16_tn_192x192x64
VARS="${P}_template ${P}_ncw3_template ${P}_1x2_h_template ${P}_1x2_v_template \
${P}_2x1_h_template ${P}_2x1_v_template ${P}_2x2_h_template ${P}_2x2_v_template \
${P}_1x4_h_template ${P}_1x4_v_template ${P}_4x1_h_template ${P}_4x1_v_template \
${P}_2x4_h_template ${P}_2x4_v_template ${P}_4x2_h_template ${P}_4x2_v_template"

echo "=== H224 cluster sweep  shape=$SIG  locked ${LOCK}MHz  $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
echo "--- correctness gate (bit-exact vs cuBLAS, independent operands) ---"
GOOD=""
for v in $VARS; do
  r=$($BIN check "$v" "$SIG" 2>&1 | tail -1)
  short=$(echo "$v" | sed "s/${P}_//; s/_template//")
  printf "  %-14s %s\n" "$short" "$(echo $r | cut -c1-70)"
  case "$r" in PASS*) GOOD="$GOOD $v";; esac
done
echo "  usable: $(echo $GOOD | wc -w) of $(echo $VARS | wc -w)"

nvidia-smi -i $GPUIDX -lgc $LOCK >/dev/null 2>&1
echo
echo "--- power + throughput, ${ROUNDS} rounds each ---"
printf "  %-14s %9s %9s %9s %9s\n" "cluster" "power" "TFLOPS" "clk" "below%"
for v in $GOOD cublas; do
  short=$(echo "$v" | sed "s/${P}_//; s/_template//")
  PS=""; TS=""
  for r in $(seq 1 $ROUNDS); do
    nvidia-smi -i $GPUIDX --query-gpu=power.draw,clocks.sm,utilization.gpu \
      --format=csv,noheader,nounits -lms 100 > /tmp/cs.csv & TELE=$!
    if [ "$v" = "cublas" ]; then tf=$($BIN cublas "$SIG" $IT 2>/dev/null | awk '{print $1}')
    else tf=$($BIN bench "$v" "$SIG" $IT 2>/dev/null | awk '{print $1}'); fi
    kill $TELE 2>/dev/null; wait $TELE 2>/dev/null
    read p c lo <<< "$(python3 -c "
import statistics
b=[(float(x[0]),float(x[1])) for x in (l.split(',') for l in open('/tmp/cs.csv') if l.strip()) if len(x)>=3 and float(x[2])>=95]
print('%.1f %.0f %.1f'%(statistics.median([y[0] for y in b]),statistics.median([y[1] for y in b]),100.0*sum(1 for y in b if y[1]<$LOCK-20)/len(b)) if b else '0 0 0')")"
    PS="$PS $p"; TS="$TS $tf"
  done
  python3 - "$short" "$PS" "$TS" "$c" "$lo" <<'PY'
import sys,statistics
s=sys.argv[1]; P=[float(x) for x in sys.argv[2].split()]; T=[float(x) for x in sys.argv[3].split() if x]
print("  %-14s %8.1fW %9.1f %9s %8s%%"%(s,statistics.median(P),statistics.median(T) if T else 0,sys.argv[4],sys.argv[5]))
PY
done
restore
echo "[clocks free]"
