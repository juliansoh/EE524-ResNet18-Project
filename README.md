# EE 524 ResNet-18 CUDA Inference Engine

**Project Team #8 | EE P 524 A/B GPU Computing | Autumn 2026**

Build a standalone CUDA C++ inference engine for pretrained ResNet-18, then
optimize it while preserving correctness. The project follows the progression
from a naive baseline through shared-memory GEMM, implicit GEMM, register tiling,
and tensor cores.

**Team target: attempt optimization tiers T0-T4.** T5 is a stretch tier outside
the initial commitment. This README describes the requirements and plan; it does
not claim that any tier has already been completed or benchmarked.

Code and project artifacts are managed in
[juliansoh/EE524-ResNet18-Project](https://github.com/juliansoh/EE524-ResNet18-Project).

## Team and responsibilities

| Member | Kernels owned | Other project tasks |
| --- | --- | --- |
| Julian Soh | Conv/GEMM path T1-T3: im2col, tiled SGEMM, implicit GEMM, register tiling, fused epilogues; executor integration. | Assembly of the final presentation and project artifacts. |
| Jasmine Soh | Tensor-core path T4: FP16 WMMA / `mma.sync` conv kernels, FP16 conversion. | Project management: milestones, scope, task tracking, submissions. |
| Colin Rocks | Normalize, maxpool, global avgpool, fully connected, and top-5 kernels. | Validation: layer and end-to-end checks, regression tests, benchmarks, NCU reports. |

T5 stretch work, the design specification, the presentation, and the final
report are shared. Owners review each other's kernels before each leaderboard snapshot.

## Assignment sources

- [Lecture introduction and assignment slides, with notes](Au26_L6_FinalProject_ResNet18-Engine_Intro_v2_w_notes.pdf):
  tier definitions on slide 15, correctness/profiling on slides 16-17, timeline
  and grading on slides 20-21, and proposal requirements on slide 22.
- [Detailed project specification](artifacts/resnet18_project/Au26_GPUCompute_ResNet18_FinalProject_Spec.pdf):
  network, data formats, engine interface, checker, submission rules, and final deliverables.
- [Starter-kit getting-started guide](artifacts/resnet18_project/GETTING_STARTED.md)
  and [engine README](artifacts/resnet18_project/starter/README.md).
- [Team proposal](EE%20524%20resNet-18%20Project%20Proposal.pdf), built from
  [proposal.html](proposal.html) with `weasyprint proposal.html "EE 524 resNet-18 Project Proposal.pdf"`.

## Deliverables and deadlines

Deadlines and grading follow the lecture assignment PDF, slides 20-21.
All dates are in **2026**; times are reproduced as stated in the assignment.

| Deliverable or event | Deadline | Grade |
| --- | --- | --- |
| Team formation/registration on Canvas | November 16, 11:59 PM* | -- |
| Starter-kit release | November 16, 9:00 AM | -- |
| Project proposal, PDF, maximum 2 pages | November 20, 11:59 PM | 10% |
| T0 smoke-test leaderboard submission | November 22, 11:59 PM | -- |
| Presentation date/time signup | November 24, 5:59 PM* | -- |
| T1+ leaderboard submissions | November 20-December 15 | -- |
| Project design specification | December 2, 11:59 PM | 21% |
| Leaderboard freeze and final project code | December 15, 5:00 PM* | 23% |
| Team presentation | December 15-17, 6:00-9:00 PM* | 23% |
| Final report, including code analysis | December 17, 11:59 PM* | 23% |

`*` Late Days cannot be used for these deadlines.

### Required content

| Deliverable | Required coverage |
| --- | --- |
| Proposal | Three team members; planned tiers; kernel/task work breakdown; specific Nsight Compute metrics and their relevance; primary references. Include the specification's requested T0 baseline, planned batch size/rationale, and tier roadmap. |
| T0 smoke test | Package and submit the unmodified starter as a valid T0 entry before optimized submissions. |
| Design specification | For each replacement kernel: grid/block/tile decomposition, layout/indexing, epilogues, arithmetic intensity and roofline predictions, memory plan, risks/fallbacks, and updated work breakdown. |
| Final code | A self-contained, buildable submission targeting `sm_75` that passes correctness. The last valid leaderboard submission before the freeze is the graded code. |
| Presentation | 10 minutes plus 5 minutes of questions: kernel design, tier-by-tier performance, roofline/profiling evidence, challenges, resolutions, and possible next steps. |
| Final report | Analyze the frozen code: kernel designs, NCU reports and rooflines per kernel/tier, tier-to-throughput ablation, before/after layer timings, remaining bottlenecks, unchanged starter kernels, and final work distribution. |

## Workload and target platform

| Property | Assignment requirement |
| --- | --- |
| Model | Pretrained ResNet-18, inference only; 20 convolutions and a 512-to-1,000 fully connected layer. |
| Input | RGB images, `224 x 224 x 3`, stored as `uint8` NHWC; normalization runs on the GPU. |
| Parameters | Provided FP32/FP16 weights with BatchNorm already folded into convolution weights and biases. No separate BatchNorm kernels. |
| Image sets | `debug`: 8 images with layer references; `dev`: 500 images; `val`: 3,925 scored images. |
| Classifier | Preserve the full 1,000-class output, although Imagenette contains 10 classes. |
| Scoring GPU | NVIDIA T4, Turing, `sm_75`, on the course AWS `g4dn.xlarge`. Development may use another compatible GPU. |
| Main score | Correctness-gated images/second, separately ranked for FP32 and FP16 reduced precision. |

There are 12 compute shape classes when the fully connected GEMM is included.
Use the manifest for graph order, dimensions, and weight offsets rather than
hard-coding assumptions. Fuse bias, optional residual addition, and optional
ReLU into convolution epilogues.

The T4 supports WMMA, `mma.sync`, warp intrinsics, and cooperative groups.
Do not rely on newer features such as `cp.async`, TMA, `wgmma`, thread-block
clusters, or FP8.

## Planned optimization tiers

Definitions follow **slide 15** of the lecture assignment.

| Tier | Definition and implementation direction |
| --- | --- |
| T0: Baseline | Build the starter, pass `check.py`, profile the 12 shape classes, select a batch size, and submit an initial leaderboard smoke test. |
| T1: Batched GEMM path | NHWC and batching; explicit im2col plus our own shared-memory-tiled SGEMM; bank-conflict-free tiles and fused bias/ReLU. |
| T2: Implicit GEMM | Remove the im2col buffer; gather input during tile loads; fuse residual addition and use per-shape launch configurations. |
| T3: Register tiling | Thread coarsening, 2-D block/register tiles, aligned `float4` loads with safe edge handling, and per-shape tile tuning. |
| T4: Tensor cores -- team target | FP16 WMMA/`mma.sync` with FP32 accumulation and FP16 weights/activations; qualify for the reduced-precision category. |
| T5: Fusion and tuning -- stretch only | Stem/maxpool fusion, shared downsample input reads, warp-shuffle average pooling/top-5, fewer launches, and per-shape autotuning. |

The lecture dates beside tiers indicate background coverage, not separate team
submission deadlines. Validate and retain evidence for each completed tier.

## Repository layout

```text
.
|-- readme.md
|-- AGENTS.md
|-- Au26_L6_FinalProject_ResNet18-Engine_Intro_v2_w_notes.pdf
|-- EE 524 resNet-18 Project Proposal.pdf
|-- proposal.html
`-- artifacts/
    |-- resnet18_project/
    |   |-- Au26_GPUCompute_ResNet18_FinalProject_Spec.pdf
    |   |-- GETTING_STARTED.md
    |   |-- REFERENCES.md
    |   |-- starter/                 # CMake project, executor, and kernels
    |   `-- tools/                   # Checker, image utilities, submission packager
    `-- resnet18_pack/
        `-- resnet18_pack/           # Local pack: manifest.json, weights, images, ref/
```

Obtain the official data pack through Canvas. The nested pack directory above
reflects the current local extraction; adjust `PACK` below if yours differs.
Point it at the directory containing `manifest.json`, not its parent.
Do not assume the large dataset binaries are available from a Git clone.

## Build and validate

### Prerequisites

- An NVIDIA CUDA-capable GPU and driver; keep T4 support (`sm_75`) in every scored build.
- CUDA Toolkit 13.x, CMake 3.18+, and a compatible C++17 compiler.
- Python with NumPy and Pillow; see the starter guide for its Python requirements.
- Nsight Compute for kernel profiling; Nsight Systems is useful for API/launch timing.

The commands below use **Bash on Linux**, starting at the repository root.
On Windows, use an x64 Visual Studio 2022 developer terminal, translate the shell
variables/path syntax, and use `starter\\build\\Release\\engine.exe` for the
Visual Studio multi-configuration build. Native CUDA/T4 performance testing
requires an NVIDIA host; it cannot run on an Apple Silicon Mac.

```bash
KIT="$PWD/artifacts/resnet18_project"
PACK="$PWD/artifacts/resnet18_pack/resnet18_pack"
ENGINE="$KIT/starter/build/engine"

python3 -m pip install numpy pillow

# Verify the official pack before trusting results.
python3 "$KIT/tools/check.py" --pack "$PACK" --verify-pack --no-e2e
sha256sum "$PACK/checksums.sha256"

# Configure and build the unmodified FP32 starter for the scoring GPU.
cmake -S "$KIT/starter" -B "$KIT/starter/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build "$KIT/starter/build" --config Release -j

# Per-layer correctness, with all eight debug images in one batch.
python3 "$KIT/tools/check.py" --pack "$PACK" --set debug \
  --out out/debug --layers --engine "$ENGINE" --batch 8

# Development-set baseline: record timing.json and correctness results.
python3 "$KIT/tools/check.py" --pack "$PACK" --set dev \
  --out out/dev --engine "$ENGINE" --batch 32 --iters 3 --warmup 1

# Full validation before submission, using the scoring iteration counts.
python3 "$KIT/tools/check.py" --pack "$PACK" --set val \
  --out out/val --engine "$ENGINE" --batch 32 --iters 5 --warmup 1
```

Expected SHA-256 of `checksums.sha256`:

```text
359370C13F34961BDF159DEC3221347A0BA605E1140B017E058FC9FCF7A38E94
```

Batch 32 is an initial example, not a measured optimum. Tune it and record the
rationale. To build for a development GPU too, retain `75` in the architecture
list, for example `-DCMAKE_CUDA_ARCHITECTURES="75;89"`. For the FP16 path,
reconfigure with `-DENABLE_FP16=ON` and pass `--precision fp16` to the checker.
Enabling FP16 storage alone does not demonstrate completion of the tensor-core tier.

### Correctness gate

| Check | FP32 | FP16 reduced precision |
| --- | --- | --- |
| Debug per-layer `max(abs(error)) / max(abs(reference))` | <= 0.001 | <= 0.02 |
| Top-1 agreement with reference logits | >= 99.5% | >= 99.5% |
| Top-1 and top-5 error versus labels | Within +/-0.5 percentage points of reference | Same |
| `top5.bin` agrees with the engine's own logits | 100% of rows | 100% of rows |
| All logits finite | Required | Required |

The full validation reference has 19.03% top-1 error and 4.03% top-5 error.
Top-5 indices must be descending by logit, with lower class indices winning ties.
A failed gate invalidates the run regardless of speed.

The engine writes `logits.bin` (FP32), `top5.bin` (int32), and `timing.json`.
Layer checks additionally use `dump/<tensor>.bin`. Preserve the supplied
command-line/output contract.

## Profiling and performance evidence

Profile T0 before optimization, then compare matching workloads at every tier.
Start with the dominant kernels and retain NCU reports, launch configurations,
batch size, precision, GPU/clock information, and before/after timings.

| Metric or analysis | Purpose |
| --- | --- |
| Global/DRAM traffic versus shared-memory accesses | Quantify tile reuse and the cost of the explicit im2col buffer. |
| `dram__bytes_read.sum`, `dram__bytes_write.sum`, DRAM throughput | Measure external-memory traffic and bandwidth pressure. |
| Shared-memory bank conflicts | Verify padding/swizzling reduces serialized accesses. |
| Global-load sectors per request | Assess coalescing; four sectors is ideal for aligned, contiguous FP32 loads by a fully active warp. |
| L1 and L2 sector hit rates | Assess activation/weight reuse alongside traffic and runtime. |
| Arithmetic intensity and roofline position | Distinguish bandwidth-bound and compute-bound kernels; use precision-appropriate FLOP counts. |
| Active warps/occupancy and warp stall reasons | Identify memory latency, MIO throttling, barriers, and resource-pressure tradeoffs. |
| Registers/thread, shared memory/block, and local-memory spills | Detect costs of larger register/shared-memory tiles. |
| Tensor-pipeline utilization | Verify tensor-core execution in the T4 optimization tier. |
| Per-layer time, images/s, and speedup | Track where improvements occur; speedup is baseline time divided by optimized time for the same workload. |

Confirm available counter names with `ncu --query-metrics` on the target GPU.
Profiling-window control through `cudaProfilerStart`/`cudaProfilerStop` with
`--profile-from-start off` is a collection method, not a hardware metric.
Use Nsight Systems if API/launch timing is needed.

Scored timing is **compute only**, from the first normalization kernel through
the final top-5 kernel across all batches; host/device transfers are excluded.
The standard scored run uses one warmup and the median of five timed iterations.
Do not use profiler-instrumented runtime as the final throughput measurement.

## Submission and implementation rules

Allowed building blocks include the CUDA runtime, streams/events, cooperative
groups, warp intrinsics, WMMA, `mma.sync`, and CUDA half conversions.

The scored path cannot use cuBLAS, cuDNN, CUTLASS, CUB, Thrust, PyTorch/framework
kernels, ONNX Runtime, TensorRT, other third-party GPU compute libraries, or
precomputed outputs. The supplied reference/checking workflow is separate from
the scored implementation.

**Important:** The grader scans submitted source and CMake files for forbidden
identifiers, including matches in comments and strings. See specification
section 8 for the exact rules. A passing build is not sufficient.

Package the **unmodified starter first** for T0:

```bash
python3 "$KIT/tools/submit.py" \
  --team "Project Team #8" \
  --members "Jasmine Soh, Colin Rocks, Julian Soh" \
  --category fp32 --batch 32 --tier 0 \
  --src "$KIT/starter" --outdir submissions
```

Keep the same team name throughout the project. Subsequent submissions must
declare the actual category, batch size, and highest completed tier. Use
`--engine-args "--conv yourmode"` when the grader must select a custom
convolution mode.

The packager includes the CMake file, source, headers, third-party headers, and
submission metadata, not build outputs or the data pack. The grader must build
from that archive alone. Follow Canvas/instructor announcements for the upload
channel and leaderboard URL; they are not specified in the supplied kit.

## Primary references

1. EE P 524 lecture recordings and notes, assignment slides, and project specification.
2. Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj,
   *Programming Massively Parallel Processors: A Hands-on Approach*, 4th edition, 2022.
3. Open-source literature and whitepapers on ResNet and CNNs, including He et al.,
   [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385),
   CVPR 2016.
4. NVIDIA [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html),
   [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/),
   and [Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/).
5. [Starter-kit references](artifacts/resnet18_project/REFERENCES.md) for model,
   dataset, GEMM optimization, and tool documentation.
