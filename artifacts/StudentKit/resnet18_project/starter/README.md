# ResNet-18 inference engine - starter

Complete, correct, deliberately naive. It passes `check.py` as shipped; every kernel in it is yours to replace.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release        # sm_75 (the T4) by default
# development GPU too: -DCMAKE_CUDA_ARCHITECTURES="75;89" (4090 / L4), "75;86" (3080 Ti), "75;90" (H100/H200), or =native
# keep 75 in the list: the grader builds for sm_75 only, and sm_80+ instructions (cp.async, bf16, m16n8k16, wgmma)
# fail at ptxas there even if they ran on your GPU. CUDA 13 targets CC 7.5 and newer.
cmake --build build --config Release -j        # Windows: the binary is build\Release\engine.exe
```
Options: `-DENABLE_FP16=ON` instantiates every kernel and the engine for `half` (Tier 4; the reduced-precision category);
`-DCPU_EMULATION=ON` compiles the .cu files as plain C++ and runs kernels serially on the CPU (no GPU; very slow; for testing).

## Run and check

`PACK` below is the folder that contains `manifest.json`, the `*.bin` files, `ref/` and `checksums.sha256` (for example
`C:\data\resnet18_pack`). Use an absolute path; it is the folder holding `manifest.json`, not its parent. On Windows the
engine is `build\Release\engine.exe`. New here? Follow `GETTING_STARTED.md` in the kit root first.

```bash
./build/engine --pack PACK --set debug --out out/debug --batch 8 --dump          # per-layer dump of the 8-image debug set
python ../tools/check.py --pack PACK --out out/debug --set debug --layers        # every tensor vs. the PyTorch reference
./build/engine --pack PACK --set dev --out out/dev --batch 64 --iters 3 --warmup 1
python ../tools/check.py --pack PACK --out out/dev --set dev                     # the leaderboard gate on the dev subset
./build/engine --pack PACK --print-graph --set debug --out out/g --batch 8       # the manifest as the executor sees it
```
`--conv direct|im2col` selects the starter's two conv paths. Add your own mode in `executor.cu` (`ConvMode`) and tell the
grader to use it with `submit.py --engine-args "--conv yourmode"`.

## Where things are

| file | role | replace it? |
|---|---|---|
| `src/kernels/conv_direct.cu` | direct conv + fused epilogue, one thread per output | yes: this is the project |
| `src/kernels/conv_im2col.cu` | im2col + MMULT-v0 GEMM + epilogue | yes: your tiled GEMM goes here first |
| `src/kernels/pool.cu`, `fc_topk.cu`, `normalize.cu` | the remaining ops | yes, when they show up in the profile |
| `include/kernels.h` | parameter structs, `to_f32`/`from_f32<T>`, `launch_*` declarations | keep the launcher signatures |
| `src/executor.cu` | buffers, batching loop, timing, outputs | only to fuse ops or re-plan buffers |
| `src/manifest.cpp` | manifest.json -> Op structs | no |
| `include/cuda_compat.h` | CUDA headers or the emulation layer, `LAUNCH(...)` | no |

Tier 0 (this starter, unmodified, direct conv) measured on the course T4: 18.5 images/s, median 212485.23 ms.

Debugging with your own PyTorch on an Ampere-or-newer GPU? Set `torch.backends.cudnn.allow_tf32 = False` first: PyTorch
runs cuDNN convolutions in TF32 by default there, which would put ~1e-3 error into *your* comparison, not into the pack.
The pack's references were produced in true fp32; `check.py` never uses torch. Your CUDA engine cannot use TF32 unless it
asks for it explicitly (cuDNN/cuBLAS are forbidden, and the T4 has no TF32 hardware), so nothing to disable in the engine.
