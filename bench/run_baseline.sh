#!/usr/bin/env bash
# Baseline sweep of the unmodified kernels: GPU on the dev set, CPU emulation on the debug set.
# usage: bench/run_baseline.sh PACK_DIR OUT_DIR
set -euo pipefail
pack=$1 out=$2
here=$(cd "$(dirname "$0")/.." && pwd)
for conv in direct im2col; do
  for b in 1 8 32 64 128; do
    # plain run for images/s; --op-times syncs after every op and costs about 7% at B=1
    "$here/engine/build/engine" --pack "$pack" --set dev --out "$out/gpu-plain/$conv/b$b" --conv $conv \
      --batch $b --warmup 1 --iters 3
    "$here/engine/build/engine" --pack "$pack" --set dev --out "$out/gpu/$conv/b$b" --conv $conv \
      --batch $b --warmup 1 --iters 3 --op-times
  done
  for b in 1 8; do
    "$here/engine/build-emu/engine" --pack "$pack" --set debug --out "$out/cpu/$conv/b$b" --conv $conv \
      --batch $b --warmup 0 --iters 1 --op-times
  done
done
