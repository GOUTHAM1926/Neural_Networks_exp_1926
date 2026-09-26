#!/bin/bash
# Library-level benchmark: the SAME program (/tmp/bench_lib) run against the OLD and the NEW
# libtensor.so in alternating order (odd rounds old->new, even rounds new->old).
#   usage: ./run_lib_bench.sh <OLD Tensor-Implementations dir> <NEW Tensor-Implementations dir> [rounds=12]
# Output: one file per run in $OUT_DIR, then the summary (summarize_lib.py).
set -u
OLD=$(cd "$1" && pwd); NEW=$(cd "$2" && pwd); ROUNDS=${3:-12}
HERE=$(cd "$(dirname "$0")" && pwd)
PROF=$(cd "$NEW/../Profiler/lib" && pwd)
OUT_DIR=${OUT_DIR:-$HOME/gautam_k/bench_lib_runs}
mkdir -p "$OUT_DIR"
[ -x /tmp/bench_lib ] || { echo "/tmp/bench_lib missing: build it first (build_bench_box.sh)"; exit 1; }
[ -f "$OLD/lib/libtensor.so" ] && [ -f "$NEW/lib/libtensor.so" ] || { echo "libtensor.so missing in OLD or NEW lib/"; exit 1; }
for r in $(seq 1 "$ROUNDS"); do
    if (( r % 2 )); then order="old new"; else order="new old"; fi
    for v in $order; do
        dir=$OLD; [ "$v" = new ] && dir=$NEW
        # the library's own folder goes first, so the loader maps exactly that libtensor.so
        LD_LIBRARY_PATH="$dir/lib:$PROF:${LD_LIBRARY_PATH:-}" /tmp/bench_lib > "$OUT_DIR/$v.r$r.txt" 2>&1 || echo "run failed: $v round $r (see $OUT_DIR/$v.r$r.txt)"
        echo "done: round $r $v"
    done
done
python3 "$HERE/summarize_lib.py" "$OUT_DIR" "$ROUNDS" "$OLD" "$NEW"
