#!/bin/bash
# 2026-10-19 (GPU 0, ~3 min): per-step timelines of OUR dense forward (vj2_TL2 / vj2_TL3 = LHOd + light stamps) and FA3's
# (fa3_tl2 / fa3_tl3), the same stamps and file layout, both in the scoreboard's forward + backward regime (1,000 iterations,
# the forwards of iterations 900 and 950 written out). nvidia-smi every 200 ms for the clock. Work doc 45.51.
# Needs build_tl_1019.sh done ("BUILD TL DONE" in /root/gautam_K/build_tl.status).
# usage: setsid nohup bash /root/gautam_K/scripts/run_tl_1019.sh > /dev/null 2>&1 < /dev/null &
GK=/root/gautam_K; S=$GK/scripts; R=$GK/sft_v2/runs; L=$R/tl_1019; mkdir -p $L
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status; }
grep -q 'BUILD TL DONE' $GK/build_tl.status 2>/dev/null || { say "ABORT: build_tl_1019.sh has not finished OK"; exit 3; }
say "start | GPU 0 $(nvidia-smi -i 0 --query-gpu=name,memory.used --format=csv,noheader)"
nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,power.draw.instant,clocks_event_reasons.sw_power_cap,temperature.gpu,utilization.gpu \
  --format=csv,noheader,nounits -lms 200 > $L/clocks.csv 2>&1 &
SMI=$!; trap "kill $SMI 2> /dev/null" EXIT
for lib in vj2_TL2 vj2_TL3; do
  bash $S/trace_1017.sh 0 $lib > /dev/null 2>&1
  say "ours $lib: $(grep -h 'trace dense' $R/trace_1017_$lib/status | tr '\n' ' ')"
done
for lvl in 2 3; do
  (cd $GK/step6 && CUDA_VISIBLE_DEVICES=0 timeout 600 $GK/benv/bin/python $S/fa3_tl_run.py $GK/fa3_tl$lvl/flash-attention/hopper $L/fa3_l$lvl > $L/fa3_l$lvl.log 2>&1)
  say "FA3 level $lvl exit=$? | $(grep -h '^TRACE\|^FA3TL' $L/fa3_l$lvl.log | tr '\n' ' ')"
done
say "TL DONE"
cd $R && tar czf $GK/tl_results.tgz tl_1019 trace_1017_vj2_TL2 trace_1017_vj2_TL3 && say "PACKED tl_results.tgz"
