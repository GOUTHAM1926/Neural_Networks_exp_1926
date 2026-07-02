#!/usr/bin/env bash
# Profile cuBLAS BF16 kernels on B300 for TN/NN/NT layouts
# Captures the actual CUDA kernel names cuBLAS picks per (M,N,K,layout)
set -uo pipefail   # no -e: don't abort if one layout's test binary exits non-zero

BUILD_DIR="${BUILD_DIR:-./build}"
OUT_DIR="${OUT_DIR:-./nsys_out}"
mkdir -p "$OUT_DIR"

for LAYOUT in tn nn nt; do
    BIN="$BUILD_DIR/test_bgemm_sm103_bf16_${LAYOUT}_cluster"
    REPORT="$OUT_DIR/profile_${LAYOUT}"

    echo "════════════════════════════════════════════════════════"
    echo "  Profiling: $LAYOUT layout"
    echo "════════════════════════════════════════════════════════"

    # NOTE: nsys will exit non-zero if the target binary does, but we WANT
    # to continue to the next layout — that's why we don't use `set -e`.
    nsys profile \
        --trace=cuda \
        --sample=none \
        --cuda-graph-trace=node \
        --force-overwrite=true \
        --output "$REPORT" \
        "$BIN" || echo "  (profile $LAYOUT exited non-zero — continuing to extract stats anyway)"

    # Extract CSVs with EXPLICIT path prefix per layout so files land inside OUT_DIR.
    # (`--output <dir>` is treated as a file prefix by nsys — so we give the full path.)
    if [ -f "${REPORT}.nsys-rep" ]; then
        nsys stats --report cuda_gpu_kern_sum --format csv \
            --output "${REPORT}" "${REPORT}.nsys-rep"

        nsys stats --report cuda_gpu_trace --format csv \
            --output "${REPORT}" "${REPORT}.nsys-rep"
    else
        echo "  (no ${REPORT}.nsys-rep — skipping stats extraction)"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Demangling kernel names in the CSV reports"
echo "════════════════════════════════════════════════════════"
for csv in "$OUT_DIR"/*.csv; do
    [ -f "$csv" ] || continue
    demangled="${csv%.csv}_demangled.csv"
    c++filt < "$csv" > "$demangled"
    echo "  → $demangled"
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Top kernels per layout (by time)"
echo "════════════════════════════════════════════════════════"
for LAYOUT in tn nn nt; do
    KERN_CSV=$(ls "$OUT_DIR"/profile_${LAYOUT}_cuda_gpu_kern_sum*_demangled.csv 2>/dev/null | head -1)
    if [ -n "$KERN_CSV" ] && [ -f "$KERN_CSV" ]; then
        echo ""
        echo "── $LAYOUT layout — top 30 kernels ──"
        head -32 "$KERN_CSV"
    fi
done

echo ""
echo "Done. Reports in: $OUT_DIR"
