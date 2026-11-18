// kernels.h - what the executor needs to know about the kernels, and the helpers the kernels share.
//
// Every op is reached through a host launcher (launch_*). The starter's launchers call deliberately naive kernels
// in src/kernels/*.cu. Replace the kernels, keep the launcher signatures, and the executor keeps working.
// Storage type T is float (fp32 build) or half (ENABLE_FP16 build); arithmetic is always fp32.
#pragma once
#include "cuda_compat.h"
#include <cstddef>
#include <cstdint>

// ------------------------------------------------------------------------------------------ parameters
enum class Epi : int { None = 0, Relu = 1, AddRelu = 2 };

struct ConvParams {
    int B, H, W, C;      // input: batch, height, width, channels (NHWC)
    int OH, OW, K;       // output: height, width, channels
    int k, s, p;         // kernel size, stride, padding
    Epi epi;             // epilogue
};
struct PoolParams {
    int B, H, W, C, OH, OW, k, s, p;
};
struct NormParams {
    float mean[3], std[3];
};

// ------------------------------------------------------------------------------------------ T <-> float
__host__ __device__ __forceinline__ float to_f32(float v) { return v; }
__host__ __device__ __forceinline__ float to_f32(half v) { return __half2float(v); }
template <typename T> __host__ __device__ __forceinline__ T from_f32(float v);
template <> __host__ __device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <> __host__ __device__ __forceinline__ half from_f32<half>(float v) { return __float2half(v); }

// ------------------------------------------------------------------------------------------ launch helpers
constexpr int kDefaultBlock = 256;
inline dim3 grid1d(long long n, int block = kDefaultBlock) {
    return dim3((unsigned)((n + block - 1) / block));
}

// ------------------------------------------------------------------------------------------ host launchers
// (defined in src/kernels/*.cu; explicitly instantiated for float, and for half when ENABLE_FP16 is set)
template <typename T> void launch_normalize(const uint8_t* in, T* out, int B, int H, int W, int C, NormParams np);
template <typename T> void launch_conv_direct(const ConvParams& p, const T* in, const T* w, const T* bias, const T* skip, T* out);
template <typename T> void launch_conv_im2col(const ConvParams& p, const T* in, const T* w, const T* bias, const T* skip, T* out, T* scratchA);
template <typename T> void launch_maxpool(const PoolParams& p, const T* in, T* out);
template <typename T> void launch_avgpool(int B, int HW, int C, const T* in, T* out);
// the fc layer writes fp32 logits in every build (the category rule: logits stay fp32); top-k reads them
template <typename T> void launch_fc(int B, int K, int N, const T* in, const T* W, const T* bias, float* out);
void launch_topk(int B, int N, int k, const float* logits, int* out);

// scratch size (elements) that launch_conv_im2col needs for a given conv shape
inline size_t im2col_scratch_elems(const ConvParams& p) {
    return (size_t)p.B * p.OH * p.OW * (size_t)p.k * p.k * p.C;
}
