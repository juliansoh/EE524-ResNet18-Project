Device: NVIDIA GeForce RTX 2080. Report: `t0-direct.ncu-rep`, 25 launches.

| # | op | kernel | grid x block | regs | time ms | SM % | mem % | DRAM % | occ % (ach/theo) | eligible warps/cyc | ld sectors/req | L1 hit % | L2 hit % | GFLOP/s | DRAM AI (FLOP/B) | top stalls (cycles/issue) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | normalize | normalize_u8_kernel | (18816, 1, 1)x(256, 1, 1) | 17 | 0.130 | 29.1 | 75.1 | 56.3 | 86.7/100 | 0.43 | 1.0 | 88.0 | 96.8 | - | - | long_scoreboard 13, lg_throttle 3 |
| 1 | conv1 | conv_direct_kernel | (100352, 1, 1)x(256, 1, 1) | 60 | 36.140 | 20.4 | 49.9 | 0.8 | 87.8/100 | 0.36 | 16.5 | 99.8 | 95.8 | 209 | 61.7 | lg_throttle 25, long_scoreboard 4 |
| 2 | maxpool | maxpool_kernel | (25088, 1, 1)x(256, 1, 1) | 34 | 0.591 | 45.2 | 50.3 | 50.3 | 85.1/100 | 0.81 | 4.0 | 22.1 | 48.1 | - | - | long_scoreboard 9, wait 3 |
| 3 | l1b0_conv1 | conv_direct_kernel | (25088, 1, 1)x(256, 1, 1) | 60 | 106.271 | 6.1 | 49.9 | 0.1 | 84.0/100 | 0.06 | 16.5 | 93.6 | 103.0 | 70 | 139.9 | lg_throttle 378, long_scoreboard 4 |
| 4 | l1b0_conv2 | conv_direct_kernel | (25088, 1, 1)x(256, 1, 1) | 60 | 106.237 | 6.1 | 49.9 | 0.2 | 87.0/100 | 0.06 | 16.5 | 94.0 | 101.7 | 70 | 93.2 | lg_throttle 391, long_scoreboard 4 |
| 5 | l1b1_conv1 | conv_direct_kernel | (25088, 1, 1)x(256, 1, 1) | 60 | 106.159 | 6.1 | 49.9 | 0.1 | 86.5/100 | 0.06 | 16.5 | 93.8 | 97.7 | 70 | 140.5 | lg_throttle 389, long_scoreboard 4 |
| 6 | l1b1_conv2 | conv_direct_kernel | (25088, 1, 1)x(256, 1, 1) | 60 | 106.205 | 6.1 | 49.9 | 0.2 | 85.9/100 | 0.06 | 16.5 | 93.8 | 101.4 | 70 | 93.2 | lg_throttle 386, long_scoreboard 4 |
| 7 | l2b0_conv1 | conv_direct_kernel | (12544, 1, 1)x(256, 1, 1) | 60 | 53.194 | 6.0 | 49.8 | 0.2 | 99.0/100 | 0.07 | 16.5 | 90.1 | 99.8 | 70 | 92.6 | lg_throttle 447, long_scoreboard 4 |
| 8 | l2b0_ds | conv_direct_kernel | (12544, 1, 1)x(256, 1, 1) | 60 | 6.080 | 6.2 | 49.9 | 0.7 | 98.3/100 | 0.08 | 16.4 | 99.7 | 84.9 | 68 | 22.0 | lg_throttle 289, wait 2 |
| 9 | l2b0_conv2 | conv_direct_kernel | (12544, 1, 1)x(256, 1, 1) | 60 | 104.116 | 6.0 | 49.7 | 0.1 | 98.9/100 | 0.07 | 16.4 | 84.3 | 99.8 | 71 | 181.4 | lg_throttle 507, long_scoreboard 5 |
| 10 | l2b1_conv1 | conv_direct_kernel | (12544, 1, 1)x(256, 1, 1) | 60 | 104.267 | 6.0 | 49.8 | 0.1 | 98.9/100 | 0.07 | 16.4 | 83.5 | 100.0 | 71 | 274.7 | lg_throttle 508, long_scoreboard 5 |
| 11 | l2b1_conv2 | conv_direct_kernel | (12544, 1, 1)x(256, 1, 1) | 60 | 104.199 | 6.0 | 49.8 | 0.1 | 98.9/100 | 0.07 | 16.4 | 84.9 | 100.0 | 71 | 181.8 | lg_throttle 507, long_scoreboard 5 |
| 12 | l3b0_conv1 | conv_direct_kernel | (6272, 1, 1)x(256, 1, 1) | 60 | 52.361 | 6.0 | 49.5 | 0.1 | 98.7/100 | 0.07 | 16.4 | 74.0 | 99.9 | 71 | 172.6 | lg_throttle 508, long_scoreboard 5 |
| 13 | l3b0_ds | conv_direct_kernel | (6272, 1, 1)x(256, 1, 1) | 60 | 6.089 | 6.1 | 49.6 | 0.3 | 98.2/100 | 0.07 | 16.4 | 82.2 | 99.8 | 68 | 44.8 | lg_throttle 387, long_scoreboard 4 |
| 14 | l3b0_conv2 | conv_direct_kernel | (6272, 1, 1)x(256, 1, 1) | 60 | 99.794 | 6.0 | 49.4 | 0.1 | 98.8/100 | 0.07 | 16.4 | 75.1 | 100.0 | 74 | 317.1 | lg_throttle 545, long_scoreboard 6 |
| 15 | l3b1_conv1 | conv_direct_kernel | (6272, 1, 1)x(256, 1, 1) | 60 | 99.865 | 6.0 | 49.4 | 0.0 | 98.8/100 | 0.06 | 16.4 | 76.6 | 100.0 | 74 | 446.1 | lg_throttle 546, long_scoreboard 5 |
| 16 | l3b1_conv2 | conv_direct_kernel | (6272, 1, 1)x(256, 1, 1) | 60 | 99.781 | 6.0 | 49.4 | 0.1 | 98.8/100 | 0.07 | 16.4 | 77.2 | 99.9 | 74 | 316.9 | lg_throttle 545, long_scoreboard 5 |
| 17 | l4b0_conv1 | conv_direct_kernel | (3136, 1, 1)x(256, 1, 1) | 60 | 50.176 | 6.0 | 49.2 | 3.4 | 97.8/100 | 0.06 | 16.4 | 76.9 | 97.3 | 74 | 5.0 | lg_throttle 539, long_scoreboard 6 |
| 18 | l4b0_ds | conv_direct_kernel | (3136, 1, 1)x(256, 1, 1) | 60 | 6.097 | 6.0 | 49.2 | 0.2 | 97.5/100 | 0.07 | 16.4 | 79.3 | 99.8 | 67 | 83.7 | lg_throttle 462, long_scoreboard 5 |
| 19 | l4b0_conv2 | conv_direct_kernel | (3136, 1, 1)x(256, 1, 1) | 60 | 90.879 | 5.9 | 49.0 | 16.8 | 97.8/100 | 0.06 | 16.4 | 77.2 | 85.5 | 81 | 1.1 | lg_throttle 560, long_scoreboard 7 |
| 20 | l4b1_conv1 | conv_direct_kernel | (3136, 1, 1)x(256, 1, 1) | 60 | 90.945 | 5.9 | 48.9 | 16.8 | 97.8/100 | 0.06 | 16.4 | 77.9 | 84.9 | 81 | 1.1 | lg_throttle 560, long_scoreboard 7 |
| 21 | l4b1_conv2 | conv_direct_kernel | (3136, 1, 1)x(256, 1, 1) | 60 | 91.158 | 5.9 | 49.0 | 17.4 | 97.8/100 | 0.06 | 16.4 | 77.6 | 85.0 | 81 | 1.1 | lg_throttle 560, long_scoreboard 7 |
| 22 | avgpool | avgpool_kernel | (64, 1, 1)x(256, 1, 1) | 44 | 0.011 | 7.3 | 76.8 | 76.8 | 34.6/100 | 0.09 | 4.0 | 0.0 | 4.4 | - | - | long_scoreboard 31, wait 2 |
| 23 | fc | fc_kernel | (125, 1, 1)x(256, 1, 1) | 47 | 0.542 | 5.4 | 44.8 | 0.9 | 50.6/100 | 0.04 | 16.5 | 82.9 | 98.5 | 60 | 15.2 | lg_throttle 265, long_scoreboard 6 |
| 24 | top5 | topk_kernel | (1, 1, 1)x(64, 1, 1) | 53 | 0.999 | 0.1 | 0.1 | 0.0 | 3.1/100 | 0.13 | 23.7 | 95.4 | 87.3 | - | - | wait 3, long_scoreboard 3 |

Sum of kernel times: 1522.29 ms for one batch of 32. GFLOP/s and AI use the exact GEMM FLOP count; DRAM AI uses measured DRAM bytes.

#### Nsight Compute guidance (longest launch of each kernel)

**normalize_u8_kernel** (launch #0, op `normalize`, 0.130 ms; all normalize_u8_kernel launches: 0.0% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 93.7% | Memory Workload Analysis Tables | The memory access pattern for local loads from L1TEX might not be optimal. |
| 93.7% | Memory Workload Analysis Tables | The memory access pattern for local stores to L1TEX might not be optimal. |
| 24.9% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 3.4 cycles. |
| 24.9% | Warp State Statistics | On average, each warp of this workload spends 12.8 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. |
| 13.3% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (86.7%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| - | GPU Speed Of Light Throughput | Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the L2 bottleneck. |
| - | PM Sampling | Sampling interval is larger than 10% of the workload duration, which likely results in very few collected samples. |

**conv_direct_kernel** (launch #3, op `l1b0_conv1`, 106.271 ms; all conv_direct_kernel launches: 99.9% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 99.0% | Compute Workload Analysis | All compute pipelines are under-utilized. |
| 87.3% | Memory Workload Analysis Tables | The memory access pattern for global loads from L1TEX might not be optimal. |
| 59.4% | Source Counters | This kernel has uncoalesced global accesses resulting in a total of 3160342528 excessive sectors (85% of the total 3726295040 sectors). |
| 50.1% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 57.6 cycles. |
| 50.1% | Warp State Statistics | On average, each warp of this workload spends 377.7 cycles being stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full. |
| 16.0% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (84.0%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| 8.4% | GPU and Memory Workload Distribution | One or more L2 Slices have a much higher number of active cycles than the average number of active cycles. |
| - | GPU Speed Of Light Throughput | This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. |

**maxpool_kernel** (launch #2, op `maxpool`, 0.591 ms; all maxpool_kernel launches: 0.0% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 49.6% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.2 cycles. |
| 49.6% | Warp State Statistics | On average, each warp of this workload spends 8.9 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. |
| 14.9% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (85.1%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| - | GPU Speed Of Light Throughput | This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. |

**avgpool_kernel** (launch #22, op `avgpool`, 0.011 ms; all avgpool_kernel launches: 0.0% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 94.1% | Compute Workload Analysis | All compute pipelines are under-utilized. |
| 23.2% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 13.8 cycles. |
| 23.2% | Warp State Statistics | On average, each warp of this workload spends 30.8 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. |
| 23.2% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (34.6%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| 6.2% | GPU and Memory Workload Distribution | One or more SMSPs have a much higher number of active cycles than the average number of active cycles. |
| 5.2% | GPU and Memory Workload Distribution | One or more SMs have a much higher number of active cycles than the average number of active cycles. |
| 5.2% | GPU and Memory Workload Distribution | One or more L1 Slices have a much higher number of active cycles than the average number of active cycles. |
| 2.7% | Instruction Statistics | This kernel executes 2560 fused and 25088 non-fused FP32 instructions. |
| - | GPU Speed Of Light Throughput | Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the DRAM bottleneck. |
| - | PM Sampling | Sampling interval is 1.2x of the workload duration, which likely results in no or very few collected samples. |
| - | Launch Statistics | If you execute __syncthreads() to synchronize the threads of a block, it is recommended to have at least two blocks per multiprocessor (compared to the currently executed 1.4 blocks) This way, blocks that aren't waiting  |

**fc_kernel** (launch #23, op `fc`, 0.542 ms; all fc_kernel launches: 0.0% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 99.1% | Compute Workload Analysis | All compute pipelines are under-utilized. |
| 78.4% | Memory Workload Analysis Tables | The memory access pattern for global loads from L1TEX might not be optimal. |
| 78.3% | Source Counters | This kernel has uncoalesced global accesses resulting in a total of 14336000 excessive sectors (85% of the total 16916288 sectors). |
| 55.2% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 68.4 cycles. |
| 55.2% | Warp State Statistics | On average, each warp of this workload spends 265.4 cycles being stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full. |
| 49.4% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (50.6%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| 12.2% | GPU and Memory Workload Distribution | One or more SMSPs have a much lower number of active cycles than the average number of active cycles. |
| 12.1% | GPU and Memory Workload Distribution | One or more SMs have a much lower number of active cycles than the average number of active cycles. |
| 12.1% | GPU and Memory Workload Distribution | One or more L1 Slices have a much lower number of active cycles than the average number of active cycles. |
| - | GPU Speed Of Light Throughput | This kernel grid is too small to fill the available resources on this device, resulting in only 0.7 full waves across all SMs. |

**topk_kernel** (launch #24, op `top5`, 0.999 ms; all topk_kernel launches: 0.1% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 97.8% | Launch Statistics | The grid for this launch is configured to execute only 1 block, which is less than the GPU's 46 multiprocessors. |
| 96.0% | Compute Workload Analysis | All compute pipelines are under-utilized. |
| 86.9% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 7.6 cycles. |
| 86.9% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (3.1%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| 44.7% | Warp State Statistics | On average, each warp of this workload spends 3.4 cycles being stalled waiting on a fixed latency execution dependency. |
| 34.7% | Warp State Statistics | On average, each warp of this workload spends 2.6 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. |
| 1.6% | Source Counters | This kernel has uncoalesced global accesses resulting in a total of 299786 excessive sectors (83% of the total 359780 sectors). |
| 0.2% | Memory Workload Analysis Tables | The memory access pattern for global loads from L1TEX might not be optimal. |
| 0.2% | Memory Workload Analysis Tables | The memory access pattern for global stores to L1TEX might not be optimal. |
| - | GPU Speed Of Light Throughput | This kernel grid is too small to fill the available resources on this device, resulting in only 0.0 full waves across all SMs. |

