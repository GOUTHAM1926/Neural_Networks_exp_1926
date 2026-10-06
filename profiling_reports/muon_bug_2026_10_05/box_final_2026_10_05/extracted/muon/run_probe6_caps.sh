#!/usr/bin/env bash
# ns_probe6 with different numel-only caps for V1n/V3n. Run run_probe6.sh first (it builds ns_probe6). GPU 2. Scratch only.
set -uo pipefail
T=$HOME/gautam_k/muon; OUT=$T/logs/probe6; export CUDA_VISIBLE_DEVICES=2
cd "$T" || exit 1
for cap in 256 512 768; do
    echo "######## cap $cap"
    ./ns_probe6 - "$cap" 2>&1 | tee "$OUT/probe6_cap$cap.txt" | grep -E "^---|V0 library|V1 fixed|V1n|speed"
done
