#!/usr/bin/env bash
# Full Nsight Compute profile of the first batch of 32 (every launch), both conv modes, as in the T4 T0 report.
# usage: bench/profile_t0.sh PACK_DIR OUT_DIR   (ncu from $NCU, else PATH)
set -euo pipefail
pack=$1 out=$2 ncu=${NCU:-ncu}
here=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"
for spec in direct:25 im2col:45; do
  conv=${spec%%:*} launches=${spec##*:}
  "$ncu" --set full --import-source yes --launch-count "$launches" -f -o "$out/t0-$conv" \
    "$here/engine/build/engine" --pack "$pack" --set dev --out "$out/run-$conv" --conv "$conv" \
    --batch 32 --warmup 0 --iters 1
done
