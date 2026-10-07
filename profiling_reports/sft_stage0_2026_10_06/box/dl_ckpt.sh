#!/usr/bin/env bash
# Download only what the weights-only load needs: metadata.bin + data_rank_0.bin (public HF repo, no token).
set -uo pipefail
D=$HOME/gautam_k/ckpt/blumodel_step_8000; mkdir -p "$D"; cd "$D" || exit 1
U=https://huggingface.co/navingv/curriculum_8btokens/resolve/main/blumodel_step_8000
echo "=== download start $(date '+%F %T')"
for f in metadata.bin data_rank_0.bin; do
    curl -sSL --fail --retry 5 -C - -o "$f" "$U/$f" || { echo "FAILED: $f"; exit 1; }
    echo "$f: $(stat -c %s "$f") bytes  md5 $(md5sum "$f" | cut -c1-32)  $(date '+%T')"
done
echo "=== download done $(date '+%F %T')"
