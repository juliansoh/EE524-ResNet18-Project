Device: Tesla T4. Report: `t0-overview.ncu-rep`, 25 launches.

| # | op | kernel | grid x block | regs | time ms | SM % | mem % | DRAM % | occ % (ach/theo) | eligible warps/cyc | ld sectors/req | L1 hit % | L2 hit % | GFLOP/s | DRAM AI (FLOP/B) | top stalls (cycles/issue) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | normalize | normalize_u8_kernel | (4704, 1, 1)x(256, 1, 1) | 17 | 0.037 | 27.7 | 61.9 | 56.7 | 86.7/100 | 0.42 | 1.0 | 88.0 | 96.8 | - | - | long_scoreboard 13, lg_throttle 3 |
| 1 | conv1 | conv_direct_kernel | (25088, 1, 1)x(256, 1, 1) | 60 | 10.064 | 20.4 | 49.9 | 0.9 | 90.8/100 | 0.37 | 16.5 | 99.8 | 96.4 | 188 | 63.6 | lg_throttle 26, long_scoreboard 4 |
| 2 | maxpool | maxpool_kernel | (6272, 1, 1)x(256, 1, 1) | 34 | 0.179 | 41.2 | 57.4 | 57.4 | 85.8/100 | 0.75 | 4.0 | 22.6 | 47.8 | - | - | long_scoreboard 10, wait 3 |
| 3 | l1b0_conv1 | conv_direct_kernel | (6272, 1, 1)x(256, 1, 1) | 60 | 29.520 | 6.0 | 49.8 | 0.1 | 91.6/100 | 0.06 | 16.5 | 93.3 | 100.7 | 63 | 149.6 | lg_throttle 412, long_scoreboard 4 |
| 4 | l1b0_conv2 | conv_direct_kernel | (6272, 1, 1)x(256, 1, 1) | 60 | 29.541 | 6.1 | 49.8 | 0.2 | 90.5/100 | 0.06 | 16.5 | 93.6 | 97.6 | 63 | 95.8 | lg_throttle 407, long_scoreboard 4 |
| 5 | l1b1_conv1 | conv_direct_kernel | (6272, 1, 1)x(256, 1, 1) | 60 | 29.569 | 6.0 | 49.8 | 0.1 | 91.0/100 | 0.06 | 16.5 | 93.4 | 100.9 | 63 | 150.6 | lg_throttle 410, long_scoreboard 4 |
| 6 | l1b1_conv2 | conv_direct_kernel | (6272, 1, 1)x(256, 1, 1) | 60 | 29.570 | 6.1 | 49.8 | 0.2 | 91.0/100 | 0.06 | 16.5 | 93.5 | 100.5 | 63 | 95.9 | lg_throttle 409, long_scoreboard 4 |
| 7 | l2b0_conv1 | conv_direct_kernel | (3136, 1, 1)x(256, 1, 1) | 60 | 14.855 | 6.0 | 49.6 | 0.2 | 97.8/100 | 0.07 | 16.5 | 89.9 | 99.7 | 62 | 93.6 | lg_throttle 441, long_scoreboard 4 |
| 8 | l2b0_ds | conv_direct_kernel | (3136, 1, 1)x(256, 1, 1) | 60 | 1.703 | 6.1 | 49.5 | 0.7 | 97.6/100 | 0.08 | 16.4 | 99.6 | 86.4 | 60 | 26.0 | lg_throttle 287, wait 2 |
| 9 | l2b0_conv2 | conv_direct_kernel | (3136, 1, 1)x(256, 1, 1) | 60 | 28.935 | 6.0 | 49.6 | 0.1 | 97.8/100 | 0.07 | 16.4 | 84.1 | 99.9 | 64 | 179.9 | lg_throttle 500, long_scoreboard 5 |
| 10 | l2b1_conv1 | conv_direct_kernel | (3136, 1, 1)x(256, 1, 1) | 60 | 29.089 | 6.0 | 49.6 | 0.1 | 97.8/100 | 0.07 | 16.4 | 82.4 | 100.0 | 64 | 284.2 | lg_throttle 502, long_scoreboard 5 |
| 11 | l2b1_conv2 | conv_direct_kernel | (3136, 1, 1)x(256, 1, 1) | 60 | 29.034 | 6.0 | 49.6 | 0.1 | 97.8/100 | 0.07 | 16.4 | 84.7 | 100.1 | 64 | 180.5 | lg_throttle 501, long_scoreboard 5 |
| 12 | l3b0_conv1 | conv_direct_kernel | (1568, 1, 1)x(256, 1, 1) | 60 | 14.697 | 5.9 | 48.9 | 0.1 | 96.2/100 | 0.06 | 16.4 | 74.2 | 99.9 | 63 | 147.2 | lg_throttle 494, long_scoreboard 5 |
| 13 | l3b0_ds | conv_direct_kernel | (1568, 1, 1)x(256, 1, 1) | 60 | 1.730 | 6.0 | 49.0 | 0.3 | 95.9/100 | 0.07 | 16.4 | 82.4 | 99.8 | 59 | 61.2 | lg_throttle 377, long_scoreboard 4 |
| 14 | l3b0_conv2 | conv_direct_kernel | (1568, 1, 1)x(256, 1, 1) | 60 | 27.909 | 6.0 | 49.3 | 0.1 | 96.4/100 | 0.06 | 16.4 | 74.8 | 99.9 | 66 | 235.1 | lg_throttle 531, long_scoreboard 6 |
| 15 | l3b1_conv1 | conv_direct_kernel | (1568, 1, 1)x(256, 1, 1) | 60 | 27.854 | 5.9 | 49.2 | 0.1 | 96.3/100 | 0.06 | 16.4 | 76.8 | 100.0 | 66 | 312.1 | lg_throttle 531, long_scoreboard 5 |
| 16 | l3b1_conv2 | conv_direct_kernel | (1568, 1, 1)x(256, 1, 1) | 60 | 27.783 | 6.0 | 49.2 | 0.1 | 96.4/100 | 0.06 | 16.4 | 77.4 | 99.9 | 67 | 235.8 | lg_throttle 531, long_scoreboard 5 |
| 17 | l4b0_conv1 | conv_direct_kernel | (784, 1, 1)x(256, 1, 1) | 60 | 14.354 | 5.8 | 48.1 | 3.2 | 92.8/100 | 0.06 | 16.4 | 76.1 | 97.5 | 64 | 6.2 | lg_throttle 508, long_scoreboard 6 |
| 18 | l4b0_ds | conv_direct_kernel | (784, 1, 1)x(256, 1, 1) | 60 | 1.717 | 5.9 | 48.8 | 0.2 | 92.5/100 | 0.06 | 16.4 | 79.6 | 100.1 | 60 | 93.9 | lg_throttle 438, long_scoreboard 5 |
| 19 | l4b0_conv2 | conv_direct_kernel | (784, 1, 1)x(256, 1, 1) | 60 | 25.844 | 5.8 | 48.0 | 15.8 | 92.8/100 | 0.06 | 16.4 | 78.0 | 88.8 | 72 | 1.4 | lg_throttle 529, long_scoreboard 7 |
| 20 | l4b1_conv1 | conv_direct_kernel | (784, 1, 1)x(256, 1, 1) | 60 | 26.247 | 5.8 | 48.3 | 15.9 | 92.9/100 | 0.06 | 16.4 | 78.1 | 88.5 | 70 | 1.4 | lg_throttle 529, long_scoreboard 7 |
| 21 | l4b1_conv2 | conv_direct_kernel | (784, 1, 1)x(256, 1, 1) | 60 | 25.833 | 5.8 | 48.0 | 15.7 | 93.0/100 | 0.06 | 16.4 | 78.0 | 88.5 | 72 | 1.4 | lg_throttle 529, long_scoreboard 7 |
| 22 | avgpool | avgpool_kernel | (16, 1, 1)x(256, 1, 1) | 44 | 0.006 | 3.6 | 45.0 | 45.0 | 23.7/100 | 0.12 | 4.0 | 0.0 | 9.0 | - | - | long_scoreboard 12, imc_miss 2 |
| 23 | fc | fc_kernel | (32, 1, 1)x(256, 1, 1) | 47 | 0.173 | 4.7 | 38.3 | 3.8 | 24.7/100 | 0.02 | 16.5 | 87.7 | 93.5 | 47 | 3.9 | lg_throttle 115, long_scoreboard 5 |
| 24 | top5 | topk_kernel | (1, 1, 1)x(64, 1, 1) | 53 | 0.845 | 0.1 | 0.1 | 0.0 | 3.1/100 | 0.15 | 6.0 | 99.0 | 48.9 | - | - | wait 3, long_scoreboard 2 |

Sum of kernel times: 427.09 ms for one batch of 8. GFLOP/s and AI use the exact GEMM FLOP count; DRAM AI uses measured DRAM bytes.

#### Nsight Compute guidance (longest launch of each kernel)

**normalize_u8_kernel** (launch #0, op `normalize`, 0.037 ms; all normalize_u8_kernel launches: 0.0% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 88.9% | Memory Workload Analysis Tables | The memory access pattern for local loads from L1TEX might not be optimal. |
| 88.9% | Memory Workload Analysis Tables | The memory access pattern for local stores to L1TEX might not be optimal. |
| 38.1% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 3.5 cycles. |
| 22.2% | Memory Workload Analysis Tables | This workload accesses local memory, accounting for 90.57% of all sectors requested in L1TEX. |
| 21.1% | Warp State Statistics | On average, each warp of this workload spends 13.2 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. |
| 13.3% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (86.7%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| 5.5% | GPU and Memory Workload Distribution | One or more DRAM Slices have a much higher number of active cycles than the average number of active cycles. |
| - | GPU Speed Of Light Throughput | Memory is more heavily utilized than Compute, dominated by L2 (61.9% of peak). |
| - | PM Sampling | Sampling interval is larger than 10% of the workload duration, which likely results in very few collected samples. |

**conv_direct_kernel** (launch #6, op `l1b1_conv2`, 29.570 ms; all conv_direct_kernel launches: 99.7% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 99.0% | Compute Workload Analysis | All compute pipelines are under-utilized. |
| 87.2% | Memory Workload Analysis Tables | The memory access pattern for global loads from L1TEX might not be optimal. |
| 55.5% | Source Counters | This workload has uncoalesced global accesses resulting in a total of 790085632 excessive sectors (85% of the total 931774464 sectors). |
| 50.2% | Scheduler Statistics | The scheduler issues an instruction only every 58 cycles on average, indicating severe issue starvation. |
| 48.9% | Warp State Statistics | On average, each warp of this workload spends 409.0 cycles being stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full. |
| 9.6% | GPU and Memory Workload Distribution | One or more L2 Slices have a much higher number of active cycles than the average number of active cycles. |
| - | GPU Speed Of Light Throughput | This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. |

**maxpool_kernel** (launch #2, op `maxpool`, 0.179 ms; all maxpool_kernel launches: 0.0% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 42.6% | Scheduler Statistics | Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.4 cycles. |
| 26.8% | Warp State Statistics | On average, each warp of this workload spends 10.2 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. |
| 14.2% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (85.8%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| - | GPU Speed Of Light Throughput | This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. |

**avgpool_kernel** (launch #22, op `avgpool`, 0.006 ms; all avgpool_kernel launches: 0.0% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 97.5% | Compute Workload Analysis | All compute pipelines are under-utilized. |
| 60.0% | Launch Statistics | The grid for this launch is configured to execute only 16 blocks, which is less than the 40 multiprocessors used. |
| 55.0% | Scheduler Statistics | The scheduler issues an instruction only every 10 cycles on average, indicating severe issue starvation. |
| 55.0% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (23.7%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| 32.5% | Warp State Statistics | On average, each warp of this workload spends 11.8 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. |
| 19.7% | GPU and Memory Workload Distribution | One or more SMs have a much lower number of active cycles than the average number of active cycles. |
| 19.7% | GPU and Memory Workload Distribution | One or more L1 Slices have a much lower number of active cycles than the average number of active cycles. |
| 19.3% | GPU and Memory Workload Distribution | One or more SMSPs have a much lower number of active cycles than the average number of active cycles. |
| 7.9% | GPU and Memory Workload Distribution | One or more L2 Slices have a much higher number of active cycles than the average number of active cycles. |
| 1.1% | Instruction Statistics | This workload executes 640 fused and 6272 non-fused FP32 instructions. |
| - | GPU Speed Of Light Throughput | This kernel grid is too small to fill the available resources on this device, resulting in only 0.10 full waves across all SMs. |
| - | PM Sampling | Sampling interval is 2.1x of the workload duration, which likely results in no or very few collected samples. |

**fc_kernel** (launch #23, op `fc`, 0.173 ms; all fc_kernel launches: 0.0% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 99.4% | Compute Workload Analysis | All compute pipelines are under-utilized. |
| 69.6% | Source Counters | This workload has uncoalesced global accesses resulting in a total of 3584000 excessive sectors (85% of the total 4229072 sectors). |
| 67.0% | Memory Workload Analysis Tables | The memory access pattern for global loads from L1TEX might not be optimal. |
| 61.7% | Scheduler Statistics | The scheduler issues an instruction only every 63 cycles on average, indicating severe issue starvation. |
| 61.7% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (24.7%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| 56.8% | Warp State Statistics | On average, each warp of this workload spends 114.6 cycles being stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full. |
| 21.8% | GPU and Memory Workload Distribution | One or more SMSPs have a much lower number of active cycles than the average number of active cycles. |
| 21.8% | GPU and Memory Workload Distribution | One or more SMs have a much lower number of active cycles than the average number of active cycles. |
| 21.8% | GPU and Memory Workload Distribution | One or more L1 Slices have a much lower number of active cycles than the average number of active cycles. |
| 20.0% | Launch Statistics | The grid for this launch is configured to execute only 32 blocks, which is less than the 40 multiprocessors used. |
| - | GPU Speed Of Light Throughput | This kernel grid is too small to fill the available resources on this device, resulting in only 0.20 full waves across all SMs. |

**topk_kernel** (launch #24, op `top5`, 0.845 ms; all topk_kernel launches: 0.2% of batch time)

| est. speedup | section | finding |
|---|---|---|
| 99.9% | Compute Workload Analysis | All compute pipelines are under-utilized. |
| 97.5% | Launch Statistics | The grid for this launch is configured to execute only 1 block, which is less than the 40 multiprocessors used. |
| 85.1% | Scheduler Statistics | The scheduler issues an instruction only every 7 cycles on average, indicating severe issue starvation. |
| 85.1% | Occupancy | The difference between calculated theoretical (100.0%) and measured achieved occupancy (3.1%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. |
| 43.2% | Warp State Statistics | On average, each warp of this workload spends 3.4 cycles being stalled waiting on a fixed latency execution dependency. |
| 0.3% | Source Counters | This workload has uncoalesced global accesses resulting in a total of 74940 excessive sectors (83% of the total 89945 sectors). |
| 0.1% | Memory Workload Analysis Tables | The memory access pattern for global loads from L1TEX might not be optimal. |
| 0.1% | Memory Workload Analysis Tables | The memory access pattern for global stores to L1TEX might not be optimal. |
| 0.1% | Warp State Statistics | Instructions are executed in warps, which are groups of 32 threads. |
| - | GPU Speed Of Light Throughput | This kernel grid is too small to fill the available resources on this device, resulting in only 0.00 full waves across all SMs. |
| - | Launch Statistics | This kernel launch is configured to execute 64 threads per block. |

