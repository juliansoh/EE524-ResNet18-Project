Device: NVIDIA GeForce RTX 2080. Report: `t0-im2col.ncu-rep`, 45 launches.

| # | op | kernel | grid x block | regs | time ms | SM % | mem % | DRAM % | occ % (ach/theo) | eligible warps/cyc | ld sectors/req | L1 hit % | L2 hit % | GFLOP/s | DRAM AI (FLOP/B) | top stalls (cycles/issue) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | normalize | normalize_u8_kernel | (18816, 1, 1)x(256, 1, 1) | 17 | 0.130 | 29.1 | 74.9 | 56.1 | 86.8/100 | 0.43 | 1.0 | 88.0 | 96.7 | - | - | long_scoreboard 13, lg_throttle 3 |
| 1 | conv1 | im2col_kernel | (230496, 1, 1)x(256, 1, 1) | 28 | 2.437 | 70.6 | 24.0 | 24.0 | 90.4/100 | 1.74 | 5.9 | 29.0 | 96.7 | - | - | wait 3, long_scoreboard 2 |
| 2 | conv1 | gemm_naive_epilogue_kernel | (100352, 1, 1)x(256, 1, 1) | 25 | 36.803 | 18.4 | 49.9 | 2.1 | 98.9/100 | 0.23 | 16.5 | 99.7 | 69.3 | 205 | 22.2 | lg_throttle 93, long_scoreboard 5 |
| 3 | maxpool | maxpool_kernel | (25088, 1, 1)x(256, 1, 1) | 34 | 0.592 | 45.3 | 50.3 | 50.3 | 85.0/100 | 0.80 | 4.0 | 22.1 | 48.1 | - | - | long_scoreboard 9, wait 3 |
| 4 | l1b0_conv1 | im2col_kernel | (225792, 1, 1)x(256, 1, 1) | 28 | 2.370 | 71.0 | 24.8 | 24.8 | 89.0/100 | 1.80 | 4.0 | 2.8 | 94.2 | - | - | wait 3, not_selected 2 |
| 5 | l1b0_conv1 | gemm_naive_epilogue_kernel | (25088, 1, 1)x(256, 1, 1) | 25 | 108.677 | 6.1 | 49.9 | 0.6 | 99.3/100 | 0.07 | 16.5 | 96.6 | 96.5 | 68 | 28.4 | lg_throttle 339, long_scoreboard 11 |
| 6 | l1b0_conv2 | im2col_kernel | (225792, 1, 1)x(256, 1, 1) | 28 | 2.370 | 71.0 | 24.9 | 24.9 | 89.0/100 | 1.80 | 4.0 | 2.8 | 94.2 | - | - | wait 3, not_selected 2 |
| 7 | l1b0_conv2 | gemm_naive_epilogue_kernel | (25088, 1, 1)x(256, 1, 1) | 25 | 108.743 | 6.1 | 49.9 | 0.6 | 99.3/100 | 0.07 | 16.5 | 96.5 | 96.5 | 68 | 25.8 | lg_throttle 338, long_scoreboard 11 |
| 8 | l1b1_conv1 | im2col_kernel | (225792, 1, 1)x(256, 1, 1) | 28 | 2.370 | 71.0 | 24.9 | 24.9 | 89.0/100 | 1.80 | 4.0 | 2.8 | 94.2 | - | - | wait 3, not_selected 2 |
| 9 | l1b1_conv1 | gemm_naive_epilogue_kernel | (25088, 1, 1)x(256, 1, 1) | 25 | 108.677 | 6.1 | 49.9 | 0.6 | 99.3/100 | 0.07 | 16.5 | 96.7 | 97.0 | 68 | 28.4 | lg_throttle 339, long_scoreboard 11 |
| 10 | l1b1_conv2 | im2col_kernel | (225792, 1, 1)x(256, 1, 1) | 28 | 2.370 | 71.0 | 24.9 | 24.9 | 89.0/100 | 1.80 | 4.0 | 2.8 | 94.2 | - | - | wait 3, not_selected 2 |
| 11 | l1b1_conv2 | gemm_naive_epilogue_kernel | (25088, 1, 1)x(256, 1, 1) | 25 | 108.693 | 6.1 | 49.9 | 0.6 | 99.3/100 | 0.07 | 16.5 | 96.6 | 96.7 | 68 | 25.8 | lg_throttle 338, long_scoreboard 11 |
| 12 | l2b0_conv1 | im2col_kernel | (56448, 1, 1)x(256, 1, 1) | 28 | 0.618 | 69.0 | 30.9 | 30.9 | 89.1/100 | 1.63 | 4.0 | 0.5 | 77.4 | - | - | wait 3, long_scoreboard 2 |
| 13 | l2b0_conv1 | gemm_naive_epilogue_kernel | (12544, 1, 1)x(256, 1, 1) | 25 | 54.460 | 6.1 | 49.9 | 0.3 | 99.3/100 | 0.07 | 16.5 | 91.6 | 99.4 | 68 | 50.6 | lg_throttle 338, long_scoreboard 12 |
| 14 | l2b0_ds | im2col_kernel | (6272, 1, 1)x(256, 1, 1) | 28 | 0.072 | 67.0 | 38.8 | 38.8 | 89.6/100 | 1.61 | 4.0 | 0.0 | 50.3 | - | - | wait 3, long_scoreboard 2 |
| 15 | l2b0_ds | gemm_naive_epilogue_kernel | (12544, 1, 1)x(256, 1, 1) | 25 | 6.044 | 6.2 | 49.9 | 0.7 | 98.2/100 | 0.08 | 16.4 | 99.7 | 84.8 | 68 | 22.0 | lg_throttle 274, long_scoreboard 9 |
| 16 | l2b0_conv2 | im2col_kernel | (112896, 1, 1)x(256, 1, 1) | 28 | 1.176 | 71.5 | 25.0 | 25.0 | 89.2/100 | 1.85 | 4.0 | 2.8 | 94.1 | - | - | wait 3, not_selected 2 |
| 17 | l2b0_conv2 | gemm_naive_epilogue_kernel | (12544, 1, 1)x(256, 1, 1) | 25 | 109.153 | 6.0 | 49.8 | 0.3 | 99.4/100 | 0.07 | 16.4 | 87.6 | 99.4 | 68 | 50.8 | lg_throttle 342, long_scoreboard 14 |
| 18 | l2b1_conv1 | im2col_kernel | (112896, 1, 1)x(256, 1, 1) | 28 | 1.176 | 71.5 | 25.0 | 25.0 | 89.2/100 | 1.85 | 4.0 | 2.7 | 94.1 | - | - | wait 3, not_selected 2 |
| 19 | l2b1_conv1 | gemm_naive_epilogue_kernel | (12544, 1, 1)x(256, 1, 1) | 25 | 109.354 | 6.0 | 49.8 | 0.3 | 99.4/100 | 0.07 | 16.4 | 86.8 | 99.4 | 68 | 55.7 | lg_throttle 342, long_scoreboard 15 |
| 20 | l2b1_conv2 | im2col_kernel | (112896, 1, 1)x(256, 1, 1) | 28 | 1.176 | 71.5 | 25.0 | 25.0 | 89.2/100 | 1.85 | 4.0 | 2.7 | 94.1 | - | - | wait 3, not_selected 2 |
| 21 | l2b1_conv2 | gemm_naive_epilogue_kernel | (12544, 1, 1)x(256, 1, 1) | 25 | 109.299 | 6.0 | 49.8 | 0.3 | 99.4/100 | 0.07 | 16.5 | 88.0 | 99.7 | 68 | 50.8 | lg_throttle 342, long_scoreboard 14 |
| 22 | l3b0_conv1 | im2col_kernel | (28224, 1, 1)x(256, 1, 1) | 28 | 0.307 | 69.5 | 30.7 | 30.7 | 89.5/100 | 1.71 | 4.0 | 0.6 | 77.1 | - | - | wait 3, long_scoreboard 2 |
| 23 | l3b0_conv1 | gemm_naive_epilogue_kernel | (6272, 1, 1)x(256, 1, 1) | 25 | 55.078 | 6.0 | 49.5 | 0.2 | 98.9/100 | 0.07 | 16.3 | 71.0 | 99.9 | 67 | 95.9 | lg_throttle 340, long_scoreboard 17 |
| 24 | l3b0_ds | im2col_kernel | (3136, 1, 1)x(256, 1, 1) | 28 | 0.038 | 65.5 | 35.4 | 35.4 | 89.0/100 | 1.62 | 4.0 | 0.0 | 50.6 | - | - | wait 3, long_scoreboard 2 |
| 25 | l3b0_ds | gemm_naive_epilogue_kernel | (6272, 1, 1)x(256, 1, 1) | 25 | 6.090 | 6.1 | 49.6 | 0.3 | 98.4/100 | 0.07 | 16.4 | 81.5 | 99.7 | 67 | 44.8 | lg_throttle 304, long_scoreboard 13 |
| 26 | l3b0_conv2 | im2col_kernel | (56448, 1, 1)x(256, 1, 1) | 28 | 0.587 | 72.0 | 24.8 | 24.8 | 89.7/100 | 1.91 | 4.0 | 2.7 | 93.9 | - | - | wait 3, not_selected 2 |
| 27 | l3b0_conv2 | gemm_naive_epilogue_kernel | (6272, 1, 1)x(256, 1, 1) | 25 | 110.306 | 6.0 | 49.4 | 0.2 | 98.9/100 | 0.07 | 16.4 | 71.7 | 99.9 | 67 | 96.9 | lg_throttle 343, long_scoreboard 17 |
| 28 | l3b1_conv1 | im2col_kernel | (56448, 1, 1)x(256, 1, 1) | 28 | 0.587 | 72.0 | 24.8 | 24.8 | 89.7/100 | 1.91 | 4.0 | 2.8 | 94.0 | - | - | wait 3, not_selected 2 |
| 29 | l3b1_conv1 | gemm_naive_epilogue_kernel | (6272, 1, 1)x(256, 1, 1) | 25 | 110.344 | 6.0 | 49.4 | 0.1 | 98.9/100 | 0.07 | 16.4 | 73.0 | 99.9 | 67 | 106.2 | lg_throttle 343, long_scoreboard 17 |
| 30 | l3b1_conv2 | im2col_kernel | (56448, 1, 1)x(256, 1, 1) | 28 | 0.587 | 72.0 | 24.8 | 24.8 | 89.8/100 | 1.91 | 4.0 | 2.7 | 93.9 | - | - | wait 3, not_selected 2 |
| 31 | l3b1_conv2 | gemm_naive_epilogue_kernel | (6272, 1, 1)x(256, 1, 1) | 25 | 110.239 | 6.0 | 49.4 | 0.2 | 98.9/100 | 0.07 | 16.4 | 74.6 | 99.8 | 67 | 96.9 | lg_throttle 343, long_scoreboard 16 |
| 32 | l4b0_conv1 | im2col_kernel | (14112, 1, 1)x(256, 1, 1) | 28 | 0.153 | 69.7 | 30.1 | 30.1 | 89.9/100 | 1.79 | 4.0 | 0.6 | 76.6 | - | - | wait 3, not_selected 2 |
| 33 | l4b0_conv1 | gemm_naive_epilogue_kernel | (3136, 1, 1)x(256, 1, 1) | 25 | 55.474 | 5.9 | 49.1 | 1.3 | 97.9/100 | 0.07 | 16.4 | 74.6 | 99.1 | 67 | 12.2 | lg_throttle 339, long_scoreboard 16 |
| 34 | l4b0_ds | im2col_kernel | (1568, 1, 1)x(256, 1, 1) | 28 | 0.020 | 62.3 | 28.8 | 28.8 | 88.0/100 | 1.64 | 4.0 | 0.0 | 51.2 | - | - | wait 3, long_scoreboard 2 |
| 35 | l4b0_ds | gemm_naive_epilogue_kernel | (3136, 1, 1)x(256, 1, 1) | 25 | 6.098 | 6.0 | 49.6 | 0.2 | 97.8/100 | 0.07 | 16.4 | 78.3 | 99.8 | 67 | 84.8 | lg_throttle 320, long_scoreboard 14 |
| 36 | l4b0_conv2 | im2col_kernel | (28224, 1, 1)x(256, 1, 1) | 28 | 0.292 | 72.0 | 24.6 | 24.6 | 89.5/100 | 1.93 | 4.0 | 2.6 | 93.7 | - | - | wait 3, not_selected 2 |
| 37 | l4b0_conv2 | gemm_naive_epilogue_kernel | (3136, 1, 1)x(256, 1, 1) | 25 | 110.933 | 5.9 | 49.1 | 2.4 | 98.0/100 | 0.07 | 16.4 | 74.6 | 98.2 | 67 | 6.5 | lg_throttle 340, long_scoreboard 17 |
| 38 | l4b1_conv1 | im2col_kernel | (28224, 1, 1)x(256, 1, 1) | 28 | 0.293 | 72.0 | 24.6 | 24.6 | 89.5/100 | 1.93 | 4.0 | 2.5 | 93.6 | - | - | wait 3, not_selected 2 |
| 39 | l4b1_conv1 | gemm_naive_epilogue_kernel | (3136, 1, 1)x(256, 1, 1) | 25 | 110.957 | 5.9 | 49.1 | 2.3 | 98.0/100 | 0.07 | 16.4 | 74.9 | 98.1 | 67 | 6.7 | lg_throttle 341, long_scoreboard 16 |
| 40 | l4b1_conv2 | im2col_kernel | (28224, 1, 1)x(256, 1, 1) | 28 | 0.294 | 72.0 | 24.6 | 24.6 | 89.5/100 | 1.93 | 4.0 | 2.6 | 93.7 | - | - | wait 3, not_selected 2 |
| 41 | l4b1_conv2 | gemm_naive_epilogue_kernel | (3136, 1, 1)x(256, 1, 1) | 25 | 110.942 | 5.9 | 49.1 | 2.1 | 97.9/100 | 0.07 | 16.4 | 75.5 | 98.4 | 67 | 7.2 | lg_throttle 341, long_scoreboard 16 |
| 42 | avgpool | avgpool_kernel | (64, 1, 1)x(256, 1, 1) | 44 | 0.011 | 7.3 | 77.0 | 77.0 | 34.2/100 | 0.09 | 4.0 | 0.0 | 4.4 | - | - | long_scoreboard 31, wait 2 |
| 43 | fc | fc_kernel | (125, 1, 1)x(256, 1, 1) | 47 | 0.542 | 5.4 | 44.8 | 0.9 | 50.5/100 | 0.04 | 16.5 | 83.0 | 99.0 | 60 | 15.1 | lg_throttle 265, long_scoreboard 6 |
| 44 | top5 | topk_kernel | (1, 1, 1)x(64, 1, 1) | 53 | 1.013 | 0.1 | 0.1 | 0.0 | 3.1/100 | 0.13 | 23.7 | 95.4 | 91.3 | - | - | wait 3, long_scoreboard 3 |

Sum of kernel times: 1667.95 ms for one batch of 32. GFLOP/s and AI use the exact GEMM FLOP count; DRAM AI uses measured DRAM bytes.

#### Nsight Compute guidance (longest launch of each kernel)

**normalize_u8_kernel** (launch #0, op `normalize`, 0.130 ms; all normalize_u8_kernel launches: 0.0% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 93.6% | Memory Workload Analysis Tables | The memory access pattern for local loads from L1TEX might not be optimal. |
| 93.6% | Memory Workload Analysis Tables | The memory access pattern for local stores to L1TEX might not be optimal. |
| 25.1% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 3.4 cycles. |
| 25.1% | Warp State Statistics | On average, each warp of this workload spends 12.9 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. |
| 13.2% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (86.8%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| - | GPU Speed Of Light Throughput | Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the L2 bottleneck. |
| - | PM Sampling | Sampling interval is larger than 10% of the workload duration, which likely results in very few collected samples. |

**im2col_kernel** (launch #1, op `conv1`, 2.437 ms; all im2col_kernel launches: 1.2% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 32.0% | Warp State Statistics | On average, each warp of this workload spends 3.3 cycles being stalled waiting on a fixed latency execution dependency. |
| 14.8% | Source Counters | This kernel has uncoalesced global accesses resulting in a total of 3885632 excessive sectors (21% of the total 18557632 sectors). |
| 7.1% | GPU and Memory Workload Distribution | One or more L2 Slices have a much higher number of active cycles than the average number of active cycles. |
| 4.3% | Memory Workload Analysis Tables | The memory access pattern for global loads from L1TEX might not be optimal. |
| - | GPU Speed Of Light Throughput | Compute is more heavily utilized than Memory: Look at the Compute Workload Analysis section to see what the compute pipelines are spending their time doing. |

**gemm_naive_epilogue_kernel** (launch #39, op `l4b1_conv1`, 110.957 ms; all gemm_naive_epilogue_kernel launches: 98.7% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 98.9% | Compute Workload Analysis | All compute pipelines are under-utilized. |
| 85.9% | Memory Workload Analysis Tables | The memory access pattern for global loads from L1TEX might not be optimal. |
| 84.0% | Source Counters | This kernel has uncoalesced global accesses resulting in a total of 3236954112 excessive sectors (85% of the total 3815182336 sectors). |
| 50.9% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 46.2 cycles. |
| 50.9% | Warp State Statistics | On average, each warp of this workload spends 340.7 cycles being stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full. |
| - | GPU Speed Of Light Throughput | This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. |

**maxpool_kernel** (launch #3, op `maxpool`, 0.592 ms; all maxpool_kernel launches: 0.0% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 49.7% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.2 cycles. |
| 49.7% | Warp State Statistics | On average, each warp of this workload spends 8.9 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. |
| 15.0% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (85.0%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| - | GPU Speed Of Light Throughput | This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. |

**avgpool_kernel** (launch #42, op `avgpool`, 0.011 ms; all avgpool_kernel launches: 0.0% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 94.2% | Compute Workload Analysis | All compute pipelines are under-utilized. |
| 23.0% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 13.8 cycles. |
| 23.0% | Warp State Statistics | On average, each warp of this workload spends 30.9 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. |
| 23.0% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (34.2%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| 6.2% | GPU and Memory Workload Distribution | One or more SMSPs have a much higher number of active cycles than the average number of active cycles. |
| 5.8% | GPU and Memory Workload Distribution | One or more L2 Slices have a much higher number of active cycles than the average number of active cycles. |
| 2.6% | Instruction Statistics | This kernel executes 2560 fused and 25088 non-fused FP32 instructions. |
| - | GPU Speed Of Light Throughput | Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the DRAM bottleneck. |
| - | PM Sampling | Sampling interval is 1.2x of the workload duration, which likely results in no or very few collected samples. |
| - | Launch Statistics | If you execute __syncthreads() to synchronize the threads of a block, it is recommended to have at least two blocks per multiprocessor (compared to the currently executed 1.4 blocks) This way, blocks that aren't waiting  |

**fc_kernel** (launch #43, op `fc`, 0.542 ms; all fc_kernel launches: 0.0% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 99.1% | Compute Workload Analysis | All compute pipelines are under-utilized. |
| 78.4% | Source Counters | This kernel has uncoalesced global accesses resulting in a total of 14336000 excessive sectors (85% of the total 16916288 sectors). |
| 78.4% | Memory Workload Analysis Tables | The memory access pattern for global loads from L1TEX might not be optimal. |
| 55.2% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 68.4 cycles. |
| 55.2% | Warp State Statistics | On average, each warp of this workload spends 265.0 cycles being stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full. |
| 49.5% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (50.5%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| 12.2% | GPU and Memory Workload Distribution | One or more SMSPs have a much lower number of active cycles than the average number of active cycles. |
| 12.1% | GPU and Memory Workload Distribution | One or more SMs have a much lower number of active cycles than the average number of active cycles. |
| 12.1% | GPU and Memory Workload Distribution | One or more L1 Slices have a much lower number of active cycles than the average number of active cycles. |
| - | GPU Speed Of Light Throughput | This kernel grid is too small to fill the available resources on this device, resulting in only 0.7 full waves across all SMs. |

**topk_kernel** (launch #44, op `top5`, 1.013 ms; all topk_kernel launches: 0.1% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 97.8% | Launch Statistics | The grid for this launch is configured to execute only 1 block, which is less than the GPU's 46 multiprocessors. |
| 96.0% | Compute Workload Analysis | All compute pipelines are under-utilized. |
| 86.9% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 7.6 cycles. |
| 86.9% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (3.1%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| 44.7% | Warp State Statistics | On average, each warp of this workload spends 3.4 cycles being stalled waiting on a fixed latency execution dependency. |
| 34.6% | Warp State Statistics | On average, each warp of this workload spends 2.6 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. |
| 1.6% | Source Counters | This kernel has uncoalesced global accesses resulting in a total of 299786 excessive sectors (83% of the total 359780 sectors). |
| 0.2% | Memory Workload Analysis Tables | The memory access pattern for global loads from L1TEX might not be optimal. |
| 0.2% | Memory Workload Analysis Tables | The memory access pattern for global stores to L1TEX might not be optimal. |
| - | GPU Speed Of Light Throughput | This kernel grid is too small to fill the available resources on this device, resulting in only 0.0 full waves across all SMs. |

