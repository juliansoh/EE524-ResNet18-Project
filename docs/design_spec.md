# ResNet-18 CUDA Inference Engine: Project Design Specification

**Team 8** (Jasmine Soh, Colin Rocks, Julian Soh), EE P 524 GPU Computing, Autumn 2026

Status: draft. Sections 4 and 5 (baseline measurements and Nsight Compute analysis) are filled in.
Sections 1-3 and 6 are outlines for the team to complete.

## 1. Summary

TODO: one paragraph per planned kernel family (conv GEMM path T1-T4, normalize, pools, fc, top-5), and the
batch size choice. Section 4.6 gives the measured reasons.

## 2. Planned kernels and pseudocode

TODO: for each kernel variant: inputs and data types, grid/block/tile decomposition, index math, GPU memory
resources (global, shared, registers, constant), course techniques (phased tiling, coarsening, register tiling,
tensor cores, privatization), epilogue, and pseudocode.

| Tier | Kernel | Replaces | Owner |
|---|---|---|---|
| T1 | im2col + SMEM-tiled SGEMM, fused bias/ReLU/residual epilogue | `gemm_naive_epilogue_kernel` | Julian |
| T2 | Implicit GEMM (gather in the tile load) | `im2col_kernel` + T1 GEMM | Julian |
| T3 | Register-tiled implicit GEMM, `float4` loads | T2 | Julian |
| T4 | FP16 WMMA / `mma.sync` implicit GEMM, fp32 accumulate | T3 (fp16 category) | Jasmine |
| T1+ | normalize, maxpool, avgpool, fc, top-5 replacements | starter kernels | Colin |

## 3. Tuning and optimization matrix

TODO: one row per configuration to test: grid/block size, SMEM tile (BM x BN x BK), register tile (TM x TN),
registers/thread, SMEM/block, weight location (global, constant, read-only cache), L1/L2 notes, per shape class.

## 4. Baseline measurements (starter kernels)

### 4.1 Method

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 (Turing, sm_75, 46 SMs, 8 GB GDDR6, 448 GB/s, 4 MB L2), driver 617.42, WSL2 |
| CPU | Intel Xeon Gold 6246 @ 3.30 GHz; the CPU reference runs on one core |
| Toolchain | CUDA 13.4 (nvcc V13.4.92), GCC 16.2, CMake 4.4, `-DCMAKE_CUDA_ARCHITECTURES=75`, Release |
| Engine | Unmodified starter kernels. The engine adds one host-side option, `--op-times` |
| Data | `dev` set (500 images) for GPU runs; `debug` set (8 images) for CPU runs |
| Runs | GPU: 1 warm-up + 3 timed iterations, median. CPU: 1 timed iteration |

**GPU timing.** `--op-times` records a CUDA event pair around every op in every batch of the timed iterations,
and writes `op_times.json` (ms per op, summed over all batches, mean over iterations). The engine synchronizes
after each op in this mode. That costs about 7% at B=1 and 0.3% at B=32, so the end-to-end table (4.3) comes
from separate plain runs without `--op-times`.

**CPU reference timing.** The kit's `-DCPU_EMULATION=ON` build compiles the same kernel sources as C++ and runs
every thread of every block serially on the host. In that build `cudaEventRecord` reads
`std::chrono::steady_clock`, so the same `--op-times` code gives host-side timings per kernel type. These CPU
versions also serve as the verification reference: the emulation build passes `check.py --layers`.
The CPU figures are single-thread and include the emulation loop overhead; they are not an optimized CPU baseline.

Reproduce:

```bash
bench/run_baseline.sh PACK results/baseline     # GPU and CPU sweeps
bench/tables.py PACK results/baseline           # the tables below
```

### 4.2 Workload: shape classes and arithmetic intensity

The 20 convolutions fall into 11 shape classes; the fc layer is the 12th GEMM shape. Every conv is a GEMM with
M = B x OH x OW, N = C_out, K = k x k x C_in.

| class | convs | GEMM MxNxK | GFLOP | % FLOP | weights MiB | im2col A MiB | AI B=1 (FLOP/B) | AI B=32 (FLOP/B) |
|---|---|---|---|---|---|---|---|---|
| conv 7x7/s2 3->64 @224->112 | 1 | 12544x64x147 | 0.236 | 6.5 | 0.04 | 7.03 | 61.3 | 61.9 |
| conv 3x3/s1 64->64 @56 | 4 | 3136x64x576 | 0.925 | 25.5 | 0.14 | 6.89 | 107 | 115 |
| conv 3x3/s2 64->128 @56->28 | 1 | 784x128x576 | 0.116 | 3.2 | 0.28 | 1.72 | 77.1 | 95.3 |
| conv 1x1/s2 64->128 @56->28 | 1 | 784x128x64 | 0.013 | 0.4 | 0.03 | 0.19 | 10.4 | 10.7 |
| conv 3x3/s1 128->128 @28 | 3 | 784x128x1152 | 0.694 | 19.1 | 0.56 | 3.45 | 139 | 212 |
| conv 3x3/s2 128->256 @28->14 | 1 | 196x256x1152 | 0.116 | 3.2 | 1.12 | 0.86 | 64.8 | 181 |
| conv 1x1/s2 128->256 @28->14 | 1 | 196x256x128 | 0.013 | 0.4 | 0.12 | 0.10 | 17.5 | 21.2 |
| conv 3x3/s1 256->256 @14 | 3 | 196x256x2304 | 0.694 | 19.1 | 2.25 | 1.72 | 79.9 | 380 |
| conv 3x3/s2 256->512 @14->7 | 1 | 49x512x2304 | 0.116 | 3.2 | 4.50 | 0.43 | 23.0 | 258 |
| conv 1x1/s2 256->512 @14->7 | 1 | 49x512x256 | 0.013 | 0.4 | 0.50 | 0.05 | 15.5 | 40.5 |
| conv 3x3/s1 512->512 @7 | 3 | 49x512x4608 | 0.694 | 19.1 | 9.00 | 0.86 | 23.8 | 411 |
| fc 512->1000 | 1 | 1x1000x512 | 0.001 | 0.0 | 1.95 | 0.0000 | 0.5 | 14.6 |

Total: 3.628 GFLOP per image. AI uses compulsory traffic only (each activation, skip and weight read or written once).

Ridge points (peak FP32 / DRAM bandwidth): T4 8.1 TFLOPS / 320 GB/s = 25 FLOP/B; RTX 2080 10.1 TFLOPS /
448 GB/s = 22.5 FLOP/B. At B=32 every 3x3 and 7x7 class sits well right of the ridge (compute-bound once data
reuse is good). The stage-2 and stage-3 1x1 downsample convs and fc stay near or below the ridge
(memory-bound); the stage-4 1x1 conv is at 40 FLOP/B, about 1.8x the ridge.

### 4.3 GPU end to end

| conv mode | B=1 img/s | B=8 img/s | B=32 img/s | B=64 img/s | B=128 img/s |
|---|---|---|---|---|---|
| direct | 23.3 | 24.7 | 24.9 | 24.6 | 24.5 |
| im2col | 20.2 | 22.1 | 22.4 | 22.4 | 22.4 |

Plain runs without --op-times. The per-op tables below come from separate --op-times runs.

### 4.4 GPU per shape class

Time per image for each class, summed over the convs in the class.

**conv=direct**

| class | ops | B=1 | B=8 | B=32 | B=64 | B=128 | % time B=32 | GFLOP/s B=32 |
|---|---|---|---|---|---|---|---|---|
| normalize | 1 | 0.021 | 0.0056 | 0.0038 | 0.0037 | 0.0036 | 0.0 | - |
| conv 7x7/s2 3->64 @224->112 | 1 | 0.957 | 0.953 | 0.952 | 0.982 | 0.999 | 2.4 | 248 |
| maxpool | 1 | 0.029 | 0.022 | 0.017 | 0.017 | 0.017 | 0.0 | - |
| conv 3x3/s1 64->64 @56 | 4 | 11.2 | 11.2 | 11.2 | 11.2 | 11.3 | 27.9 | 82.6 |
| conv 3x3/s2 64->128 @56->28 | 1 | 1.482 | 1.408 | 1.402 | 1.403 | 1.412 | 3.5 | 82.4 |
| conv 1x1/s2 64->128 @56->28 | 1 | 0.186 | 0.163 | 0.160 | 0.160 | 0.161 | 0.4 | 80.2 |
| conv 3x3/s1 128->128 @28 | 3 | 8.473 | 8.250 | 8.234 | 8.244 | 8.295 | 20.5 | 84.2 |
| conv 3x3/s2 128->256 @28->14 | 1 | 1.534 | 1.391 | 1.380 | 1.380 | 1.391 | 3.4 | 83.8 |
| conv 1x1/s2 128->256 @28->14 | 1 | 0.200 | 0.164 | 0.161 | 0.160 | 0.162 | 0.4 | 79.8 |
| conv 3x3/s1 256->256 @14 | 3 | 8.312 | 7.937 | 7.884 | 7.887 | 7.980 | 19.7 | 88.0 |
| conv 3x3/s2 256->512 @14->7 | 1 | 1.581 | 1.348 | 1.321 | 1.318 | 1.346 | 3.3 | 87.5 |
| conv 1x1/s2 256->512 @14->7 | 1 | 0.208 | 0.167 | 0.161 | 0.161 | 0.164 | 0.4 | 79.6 |
| conv 3x3/s1 512->512 @7 | 3 | 8.225 | 7.359 | 7.200 | 7.263 | 7.450 | 17.9 | 96.3 |
| avgpool | 1 | 0.029 | 0.0032 | 0.0010 | 0.0006 | 0.0005 | 0.0 | - |
| fc 512->1000 | 1 | 0.095 | 0.021 | 0.015 | 0.015 | 0.014 | 0.0 | 69.6 |
| topk | 1 | 0.710 | 0.093 | 0.027 | 0.014 | 0.0072 | 0.1 | - |

**conv=im2col** (the conv rows include both `im2col_kernel` and `gemm_naive_epilogue_kernel`)

| class | ops | B=1 | B=8 | B=32 | B=64 | B=128 | % time B=32 | GFLOP/s B=32 |
|---|---|---|---|---|---|---|---|---|
| normalize | 1 | 0.023 | 0.0060 | 0.0041 | 0.0037 | 0.0036 | 0.0 | - |
| conv 7x7/s2 3->64 @224->112 | 1 | 1.069 | 1.059 | 1.068 | 1.072 | 1.083 | 2.4 | 221 |
| maxpool | 1 | 0.037 | 0.020 | 0.018 | 0.017 | 0.017 | 0.0 | - |
| conv 3x3/s1 64->64 @56 | 4 | 12.1 | 12.0 | 11.9 | 11.8 | 11.8 | 26.6 | 77.9 |
| conv 3x3/s2 64->128 @56->28 | 1 | 1.556 | 1.486 | 1.464 | 1.463 | 1.462 | 3.3 | 79.0 |
| conv 1x1/s2 64->128 @56->28 | 1 | 0.201 | 0.168 | 0.163 | 0.163 | 0.162 | 0.4 | 78.8 |
| conv 3x3/s1 128->128 @28 | 3 | 9.263 | 8.906 | 8.811 | 8.795 | 8.787 | 19.7 | 78.7 |
| conv 3x3/s2 128->256 @28->14 | 1 | 1.695 | 1.499 | 1.480 | 1.468 | 1.469 | 3.3 | 78.1 |
| conv 1x1/s2 128->256 @28->14 | 1 | 0.216 | 0.168 | 0.165 | 0.163 | 0.164 | 0.4 | 77.8 |
| conv 3x3/s1 256->256 @14 | 3 | 10.1 | 9.000 | 8.942 | 8.945 | 8.929 | 20.0 | 77.6 |
| conv 3x3/s2 256->512 @14->7 | 1 | 1.757 | 1.515 | 1.494 | 1.487 | 1.485 | 3.3 | 77.4 |
| conv 1x1/s2 256->512 @14->7 | 1 | 0.222 | 0.170 | 0.166 | 0.165 | 0.165 | 0.4 | 77.3 |
| conv 3x3/s1 512->512 @7 | 3 | 10.5 | 9.121 | 9.038 | 8.994 | 8.990 | 20.2 | 76.7 |
| avgpool | 1 | 0.031 | 0.0030 | 0.0009 | 0.0006 | 0.0004 | 0.0 | - |
| fc 512->1000 | 1 | 0.097 | 0.021 | 0.015 | 0.015 | 0.014 | 0.0 | 68.1 |
| topk | 1 | 0.725 | 0.095 | 0.028 | 0.014 | 0.0071 | 0.1 | - |

### 4.5 CPU reference

| class | direct B=1 | direct B=8 | im2col B=1 | im2col B=8 | GPU speedup (direct, B=1) |
|---|---|---|---|---|---|
| normalize | 1.172 | 1.285 | 1.200 | 1.287 | 55.25 |
| conv 7x7/s2 3->64 @224->112 | 144 | 141 | 189 | 195 | 150.57 |
| maxpool | 10.8 | 11.1 | 10.9 | 11.5 | 375.02 |
| conv 3x3/s1 64->64 @56 | 453 | 461 | 796 | 802 | 40.55 |
| conv 3x3/s2 64->128 @56->28 | 56.8 | 56.7 | 76.6 | 79.8 | 38.35 |
| conv 1x1/s2 64->128 @56->28 | 7.712 | 7.874 | 7.146 | 7.385 | 41.53 |
| conv 3x3/s1 128->128 @28 | 331 | 328 | 464 | 470 | 39.05 |
| conv 3x3/s2 128->256 @28->14 | 56.1 | 55.1 | 67.0 | 67.4 | 36.55 |
| conv 1x1/s2 128->256 @28->14 | 7.009 | 7.061 | 6.192 | 6.403 | 35.12 |
| conv 3x3/s1 256->256 @14 | 312 | 313 | 409 | 409 | 37.59 |
| conv 3x3/s2 256->512 @14->7 | 52.4 | 52.1 | 62.4 | 62.2 | 33.11 |
| conv 1x1/s2 256->512 @14->7 | 6.677 | 6.707 | 6.275 | 6.304 | 32.16 |
| conv 3x3/s1 512->512 @7 | 285 | 283 | 376 | 375 | 34.61 |
| avgpool | 0.030 | 0.028 | 0.031 | 0.031 | 1.05 |
| fc 512->1000 | 0.531 | 0.493 | 0.533 | 0.497 | 5.61 |
| topk | 0.013 | 0.011 | 0.013 | 0.012 | 0.02 |
| total | 1,724 | 1,724 | 2,473 | 2,493 | 39.9 |

### 4.6 Findings

1. **Convolution is 99.8% of the time.** The four 3x3 stride-1 classes alone take 86% (direct, B=32).
   Normalize, pools, fc and top-5 together take 0.1-0.2%.
2. **The starter conv reaches 80-96 GFLOP/s, about 1% of the 2080 FP32 peak.** Every class except the stem
   (248 GFLOP/s, C_in = 3) reaches the same rate, so the limit is the access pattern, not the shape
   (see section 5).
3. **Batch size barely changes the naive kernels.** Throughput rises from 23.3 to 24.9 images/s between B=1
   and B=32, then stays flat. The planned GEMM kernels need a large M to fill the SMs in stage 4
   (M = 49 per image) and to reuse weights: at B=32 the stage-4 AI rises from 24 to 411 FLOP/B. We use
   **B=32** as the starting point and tune it again per tier.
4. **Explicit im2col is 10% slower than direct** (22.4 vs 24.9 images/s at B=32). The naive GEMM has the same
   uncoalesced pattern as direct, and it also writes and reads the A matrix (up to 7 MiB per image per conv).
5. **The GPU top-5 is slower than the CPU at small batch.** At B=1 the kernel takes 0.71 ms per image on the
   GPU and 0.013 ms on the CPU: one thread per image does five serial scans of 1,000 logits in a 1-block grid.
6. **The GPU is about 40x faster than the serial CPU reference end to end** (1.72 s vs 43 ms per image at B=1).

## 5. Nsight Compute analysis of the starter kernels

### 5.1 Method

| Report | GPU | Batch | Conv mode | Launches | Owner |
|---|---|---|---|---|---|
| `T0/Team-8_tier0/Final-project-T0-Profile.ncu-repz` (branch `Julian-T1`) | Tesla T4 | 8 | direct | 25 | Julian |
| `T0/rtx2080/t0-direct-b32.ncu-repz` | RTX 2080 | 32 | direct | 25 | Colin |
| `T0/rtx2080/t0-im2col-b32.ncu-repz` | RTX 2080 | 32 | im2col | 45 | Colin |

Each report holds every kernel launch of the first batch, collected with `--set full --import-source yes`
(`bench/profile_t0.sh`). The `.ncu-repz` files are zstd-compressed `.ncu-rep` files; decompress with `zstd -d` if
your Nsight Compute version cannot open them. NCU locks the SM clock near base (about 1.5 GHz on the 2080), so
NCU durations are longer than the event timings in section 4. The tables below come from `bench/ncu_tables.py`.
FLOP rates use the exact GEMM FLOP count from the manifest; DRAM AI uses measured DRAM bytes.

### 5.2 `conv_direct.cu`

One launch per shape class (the other launches in a class give the same figures within 1%):

| shape class | op | T4 ms (B=8) | 2080 ms (B=32) | SM % | DRAM % | occ % | eligible warps/cyc | ld sectors/req | L1 hit % | GFLOP/s | top stall (cycles/issue) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7x7/s2 3->64 @224->112 | conv1 | 10.064 | 36.140 | 20.4 | 0.8 | 87.8 | 0.36 | 16.5 | 99.8 | 209 | lg_throttle 25 |
| 3x3/s1 64->64 @56 | l1b0_conv1 | 29.520 | 106.271 | 6.1 | 0.1 | 84.0 | 0.06 | 16.5 | 93.6 | 70 | lg_throttle 378 |
| 3x3/s2 64->128 @56->28 | l2b0_conv1 | 14.855 | 53.194 | 6.0 | 0.2 | 99.0 | 0.07 | 16.5 | 90.1 | 70 | lg_throttle 447 |
| 1x1/s2 64->128 @56->28 | l2b0_ds | 1.703 | 6.080 | 6.2 | 0.7 | 98.3 | 0.08 | 16.4 | 99.7 | 68 | lg_throttle 289 |
| 3x3/s1 128->128 @28 | l2b0_conv2 | 28.935 | 104.116 | 6.0 | 0.1 | 98.9 | 0.07 | 16.4 | 84.3 | 71 | lg_throttle 507 |
| 3x3/s2 128->256 @28->14 | l3b0_conv1 | 14.697 | 52.361 | 6.0 | 0.1 | 98.7 | 0.07 | 16.4 | 74.0 | 71 | lg_throttle 508 |
| 1x1/s2 128->256 @28->14 | l3b0_ds | 1.730 | 6.089 | 6.1 | 0.3 | 98.2 | 0.07 | 16.4 | 82.2 | 68 | lg_throttle 387 |
| 3x3/s1 256->256 @14 | l3b0_conv2 | 27.909 | 99.794 | 6.0 | 0.1 | 98.8 | 0.07 | 16.4 | 75.1 | 74 | lg_throttle 545 |
| 3x3/s2 256->512 @14->7 | l4b0_conv1 | 14.354 | 50.176 | 6.0 | 3.4 | 97.8 | 0.06 | 16.4 | 76.9 | 74 | lg_throttle 539 |
| 1x1/s2 256->512 @14->7 | l4b0_ds | 1.717 | 6.097 | 6.0 | 0.2 | 97.5 | 0.07 | 16.4 | 79.3 | 67 | lg_throttle 462 |
| 3x3/s1 512->512 @7 | l4b0_conv2 | 25.844 | 90.879 | 5.9 | 16.8 | 97.8 | 0.06 | 16.4 | 77.2 | 81 | lg_throttle 560 |

Nsight Compute guidance, longest launch (RTX 2080, B=32):

**conv_direct_kernel** (launch #3, op `l1b0_conv1`, 106.271 ms; all conv_direct_kernel launches: 99.9% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 99.0% | Compute Workload Analysis | All compute pipelines are under-utilized. |
| 87.3% | Memory Workload Analysis Tables | The memory access pattern for global loads from L1TEX might not be optimal. |
| 59.4% | Source Counters | This kernel has uncoalesced global accesses resulting in a total of 3160342528 excessive sectors (85% of the total 3726295040 sectors). |
| 50.1% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 57.6 cycles. |
| 50.1% | Warp State Statistics | On average, each warp of this workload spends 377.7 cycles being stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full. |

The T4 report gives the same findings with the same order and almost the same estimates: compute pipelines
under-utilized (99.0%), uncoalesced global loads (87.2%), excessive sectors (55.5%), issue starvation (50.2%),
LG throttle (48.9%).

**Analysis.** One thread computes one output element, and neighboring threads in a warp compute neighboring
output channels `co`. In the inner loop every lane reads the same input value (1 sector, a broadcast) and its
own weight row `w[co][kh][kw][ci]`, which is `k*k*C` floats from its neighbor (32 sectors). The mean is
(1 + 32) / 2 = 16.5 sectors per request. NCU measures 16.4-16.5 for every launch (the coalesced bias and skip
loads lower the mean slightly). Each multiply-add needs
two global loads and gets no reuse from shared memory or registers, so the warps wait on the LG queue
(289-560 cycles per issued instruction outside the stem) and only 0.06 warps per cycle are eligible. Occupancy is already
84-99%, so more warps would not help. In stages 1-3 DRAM traffic is low (0.1-0.7% of peak) because L1 and
L2 catch the reuse. In stage 4 the 9 MiB of 512->512 weights do not fit in the 4 MB L2, so DRAM rises to 17%
and the measured DRAM AI falls to about 1 FLOP/B. The T1 tile plan must reuse each weight tile across the whole
batch in that stage. Overall the kernel is latency-bound at about 6% SM throughput and 67-81 GFLOP/s. The stem class uses the same access pattern but
reaches 209 GFLOP/s, with 20% SM throughput and a 25-cycle LG stall.

### 5.3 `conv_im2col.cu`

| shape class | op | im2col ms | im2col DRAM % | im2col sectors/req | GEMM ms | GEMM GFLOP/s | GEMM sectors/req | GEMM top stall | im2col share % |
|---|---|---|---|---|---|---|---|---|---|
| 7x7/s2 3->64 @224->112 | conv1 | 2.437 | 24.0 | 5.9 | 36.803 | 205 | 16.5 | lg_throttle 93 | 6.2 |
| 3x3/s1 64->64 @56 | l1b0_conv1 | 2.370 | 24.8 | 4.0 | 108.677 | 68 | 16.5 | lg_throttle 339 | 2.1 |
| 3x3/s2 64->128 @56->28 | l2b0_conv1 | 0.618 | 30.9 | 4.0 | 54.460 | 68 | 16.5 | lg_throttle 338 | 1.1 |
| 1x1/s2 64->128 @56->28 | l2b0_ds | 0.072 | 38.8 | 4.0 | 6.044 | 68 | 16.4 | lg_throttle 274 | 1.2 |
| 3x3/s1 128->128 @28 | l2b0_conv2 | 1.176 | 25.0 | 4.0 | 109.153 | 68 | 16.4 | lg_throttle 342 | 1.1 |
| 3x3/s2 128->256 @28->14 | l3b0_conv1 | 0.307 | 30.7 | 4.0 | 55.078 | 67 | 16.3 | lg_throttle 340 | 0.6 |
| 1x1/s2 128->256 @28->14 | l3b0_ds | 0.038 | 35.4 | 4.0 | 6.090 | 67 | 16.4 | lg_throttle 304 | 0.6 |
| 3x3/s1 256->256 @14 | l3b0_conv2 | 0.587 | 24.8 | 4.0 | 110.306 | 67 | 16.4 | lg_throttle 343 | 0.5 |
| 3x3/s2 256->512 @14->7 | l4b0_conv1 | 0.153 | 30.1 | 4.0 | 55.474 | 67 | 16.4 | lg_throttle 339 | 0.3 |
| 1x1/s2 256->512 @14->7 | l4b0_ds | 0.020 | 28.8 | 4.0 | 6.098 | 67 | 16.4 | lg_throttle 320 | 0.3 |
| 3x3/s1 512->512 @7 | l4b0_conv2 | 0.292 | 24.6 | 4.0 | 110.933 | 67 | 16.4 | lg_throttle 340 | 0.3 |

Nsight Compute guidance, longest launch of each kernel (RTX 2080, B=32):

**gemm_naive_epilogue_kernel** (launch #39, op `l4b1_conv1`, 110.957 ms; all gemm_naive_epilogue_kernel launches: 98.7% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 98.9% | Compute Workload Analysis | All compute pipelines are under-utilized. |
| 85.9% | Memory Workload Analysis Tables | The memory access pattern for global loads from L1TEX might not be optimal. |
| 84.0% | Source Counters | This kernel has uncoalesced global accesses resulting in a total of 3236954112 excessive sectors (85% of the total 3815182336 sectors). |
| 50.9% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 46.2 cycles. |
| 50.9% | Warp State Statistics | On average, each warp of this workload spends 340.7 cycles being stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full. |

**im2col_kernel** (launch #1, op `conv1`, 2.437 ms; all im2col_kernel launches: 1.2% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 32.0% | Warp State Statistics | On average, each warp of this workload spends 3.3 cycles being stalled waiting on a fixed latency execution dependency. |
| 14.8% | Source Counters | This kernel has uncoalesced global accesses resulting in a total of 3885632 excessive sectors (21% of the total 18557632 sectors). |
| 7.1% | GPU and Memory Workload Distribution | One or more L2 Slices have a much higher number of active cycles than the average number of active cycles. |
| 4.3% | Memory Workload Analysis Tables | The memory access pattern for global loads from L1TEX might not be optimal. |

**Analysis.** The naive GEMM has the same defect as direct conv: lanes take neighboring `n`, so each lane reads
its own row of W (stride `Kdim`) while the A element is a broadcast, again 16.4-16.5 sectors per request and
the same LG-throttle stall, at 67-68 GFLOP/s. The `im2col_kernel` itself is cheap here: coalesced
(4 sectors per request), 24-39% of DRAM
bandwidth, and only 0.3-6% of each conv's time. That share is small only because the GEMM is slow. At B=32 the
stage-1 A matrix is 32 x 3136 x 576 x 4 B = 231 MB per conv. To write it and read it back costs about 1.4 ms
at the T4's 320 GB/s, and a GEMM at 5 TFLOPS needs 1.5 ms for the same conv. Once T1 fixes the GEMM,
im2col becomes a large fraction of the time, which is the case for implicit GEMM in T2.

### 5.4 Other starter kernels

These kernels take less than 0.2% of the time together, so they are not on the critical path. The NCU findings
still show what to fix when we replace them.

| Kernel | Time share | Main NCU finding (RTX 2080, B=32) | Cause | Planned fix |
|---|---|---|---|---|
| `normalize_u8_kernel` | <0.01% | Local memory is 90% of L1 sectors (T4); uncoalesced local loads/stores, est. 93.7% | `np.mean[c]` and `np.std[c]` index a by-value kernel parameter with a runtime index, so the compiler copies the struct to local memory (the SASS has `STL`/`LDL`; no other kernel does) | Select the channel constants without a dynamic index, or put them in `__constant__` memory; later fuse into the stem conv load |
| `maxpool_kernel` | <0.1% | Scheduler issues every 2.2 cycles, est. 49.6%; L1 hit 22% | Each input is read by up to 4 threads (2.25 on average), all from global memory | Fuse into the stem conv epilogue (T5), or tile in shared memory |
| `avgpool_kernel` | <0.01% | 64 blocks for 46 SMs (1.4 per SM), achieved occupancy 34%, est. 23% | One thread per (image, channel) with a serial 49-element loop | One warp per (image, channel group) with a warp-shuffle reduction |
| `fc_kernel` | <0.1% | Uncoalesced loads (85% excess sectors), est. 78.4%; 0.7 waves | Same pattern as the naive GEMM: lanes read different W rows | Reuse the T1 tiled GEMM with M = batch, or a warp-per-output dot product with coalesced K loads |
| `topk_kernel` | 0.1% | 1 block on 46 SMs, achieved occupancy 3.1%, est. 97.8% | One thread per image does 5 serial scans of 1,000 logits | One warp (or block) per image: per-lane top-5, then a warp-shuffle merge with the lower-index tie rule |

### 5.5 Guidance summary and planned response

| NCU finding (all conv and GEMM launches) | Est. speedup | Planned response | Tier |
|---|---|---|---|
| Uncoalesced global loads, 16.5 sectors/request, 85% excess sectors | 56-87% | Load A and W tiles cooperatively with consecutive lanes on consecutive K, so each warp request is 4 sectors | T1 |
| LG throttle: two global loads per FMA | 49-51% | Stage tiles in shared memory and reuse each loaded value BM or BN times; pad tiles to avoid bank conflicts | T1 |
| Issue starvation: 0.06 eligible warps per cycle | 50% | Raise arithmetic per load: register tiles (TM x TN outputs per thread), `float4` loads | T3 |
| All compute pipelines under-utilized (6% SM throughput) | 99% | The result of the items above; then FP16 tensor cores for the reduced-precision category | T1-T4 |
| im2col traffic grows in relative cost once the GEMM is fast | - | Gather the A tile in the GEMM tile load (implicit GEMM) | T2 |

### Appendix: full per-launch NCU tables

The full tables (every launch, every guidance item) are in `results/ncu_t4_direct_b8.md`,
`results/ncu_2080_direct_b32.md` and `results/ncu_2080_im2col_b32.md`.

## 6. Work breakdown

TODO: update from the proposal if it changed.
