#!/usr/bin/env bash
# §H227 driver: 128x256 across all 10 production shapes it serves, all 3 layouts.
# TARGET_S=6 comes from tools/h227_rampcal.sh: power plateaus at ~4 s on this rig and
# starts drifting thermally past ~8 s. A 1 s window read cuBLAS 1.28x below our kernels,
# which is a ramp artefact, not a result.
set -u
cd ~/gautam_k/blublas
for L in tn nn nt; do
  echo
  echo "################ LAYOUT $L  $(TZ=Asia/Kolkata date +%H:%M:%S) IST ################"
  PREFIX=blublasBgemmSM90_bf16_${L}_128x256x64 \
  CASES=tools/h227_cases_128x256_${L}.txt \
  OUT=h227_out_${L} TARGET_S=6 VTMO=120 \
  bash tools/h227_tile_sweep.sh
done
echo
echo "################ ALL LAYOUTS DONE  $(TZ=Asia/Kolkata date +%H:%M:%S) IST ################"
