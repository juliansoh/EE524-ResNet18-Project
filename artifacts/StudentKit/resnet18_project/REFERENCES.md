# References

Sources used to define the project and build the kit. Primary URLs verified 2026-11-09.

## Model and weights
- K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR 2016. https://arxiv.org/abs/1512.03385 (DOI https://doi.org/10.48550/arXiv.1512.03385)
- torchvision `resnet18` and `ResNet18_Weights.IMAGENET1K_V1` (weights, accuracy, inference transform):
  https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- torchvision ResNet source (module names used for `torch_name`): https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
- BatchNorm folding: `torch.nn.utils.fusion.fuse_conv_bn_eval`:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.fusion.fuse_conv_bn_eval.html

## Dataset
- J. Howard. Imagenette (fast.ai, 2019, Apache-2.0). https://github.com/fastai/imagenette
  The project standardizes on the full-size release: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz (export_pack.py --download).
  v2 (6 Dec 2019) uses a 70/30 train/val split: 9,469 train / 3,925 val images. The README does not state whether the
  val images come from ImageNet's train split; treat the pretrained model's reference accuracy as possibly optimistic.

## Target hardware
- NVIDIA T4 datasheet: https://nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
  (8.1 TFLOPS FP32, 65 TFLOPS FP16 tensor, 130 TOPS INT8, 16 GB GDDR6, 70 W; NVIDIA lists 300 GB/s, partners list 320 GB/s)
- AWS g4dn instances: https://aws.amazon.com/ec2/instance-types/g4/
- CUDA GPU compute capabilities: https://developer.nvidia.com/cuda-gpus
- Turing Tuning Guide: https://docs.nvidia.com/cuda/turing-tuning-guide/

## CUDA documentation
- CUDA C++ Programming Guide (incl. Warp Matrix Functions / WMMA): https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- PTX ISA (warp-level matrix multiply-accumulate, mma.sync): https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA Math API, half precision intrinsics: https://docs.nvidia.com/cuda/cuda-math-api/
- CUDA Runtime API (events, device properties): https://docs.nvidia.com/cuda/cuda-runtime-api/
- Nsight Compute Kernel Profiling Guide (metrics and sections): https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- nvidia-smi: https://developer.nvidia.com/system-management-interface

## Optimization progression
- S. Boehm. How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog (2022). https://siboehm.com/articles/22/CUDA-MMM
- S. Boehm. SGEMM_CUDA (kernels 1-12). https://github.com/siboehm/SGEMM_CUDA
- W. Hwu, D. Kirk, I. El Hajj. Programming Massively Parallel Processors, 4th ed. Morgan Kaufmann, 2022.

## Software used by the kit
- nlohmann/json v3.11.3 (MIT), vendored at starter/third_party/nlohmann/json.hpp: https://github.com/nlohmann/json
- CMake CUDA support: https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
- NumPy https://numpy.org/ ; Pillow https://python-pillow.org/ ; ReportLab https://www.reportlab.com/ ; PptxGenJS https://github.com/gitbrent/PptxGenJS
