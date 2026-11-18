// conv_im2col.cu - explicit im2col lowering followed by a naive GEMM with the fused epilogue (Tier 1 reference).
//   A[M][Kdim]  with M = B*OH*OW (one row per output pixel) and Kdim = k*k*C in (kh, kw, ci) order,
//   W[N][Kdim]  is the weight blob as stored ([co][kh][kw][ci]), so out[m][n] = sum_k A[m][k] * W[n][k].
// Both kernels are one-thread-per-element and read global memory only. The GEMM is MMULT v0 from L3;
// your SMEM-tiled / register-tiled GEMM drops in here first, and implicit GEMM (Tier 2) removes A altogether.
#include "kernels.h"

template <typename T>
__global__ void im2col_kernel(ConvParams p, const T* __restrict__ in, T* __restrict__ A) {
    long long Kdim = (long long)p.k * p.k * p.C;
    long long M = (long long)p.B * p.OH * p.OW;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= M * Kdim) return;
    long long m = i / Kdim, kk = i % Kdim;
    int ci = (int)(kk % p.C);
    int kw = (int)((kk / p.C) % p.k);
    int kh = (int)(kk / ((long long)p.C * p.k));
    int ox = (int)(m % p.OW);
    int oy = (int)((m / p.OW) % p.OH);
    int b = (int)(m / ((long long)p.OW * p.OH));
    int iy = oy * p.s + kh - p.p, ix = ox * p.s + kw - p.p;
    bool valid = (iy >= 0 && iy < p.H && ix >= 0 && ix < p.W);
    A[i] = valid ? in[(((long long)b * p.H + iy) * p.W + ix) * p.C + ci] : from_f32<T>(0.0f);
}

template <typename T>
__global__ void gemm_naive_epilogue_kernel(long long M, int N, long long Kdim, const T* __restrict__ A, const T* __restrict__ W,
                                           const T* __restrict__ bias, const T* __restrict__ skip, T* __restrict__ out, Epi epi) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    int n = (int)(i % N);
    long long m = i / N;
    const T* a = A + m * Kdim;
    const T* w = W + (long long)n * Kdim;
    float acc = to_f32(bias[n]);
    for (long long kk = 0; kk < Kdim; ++kk) acc += to_f32(a[kk]) * to_f32(w[kk]);
    if (epi == Epi::AddRelu) acc += to_f32(skip[i]);
    if (epi != Epi::None) acc = fmaxf(acc, 0.0f);
    out[i] = from_f32<T>(acc);
}

template <typename T>
void launch_conv_im2col(const ConvParams& p, const T* in, const T* w, const T* bias, const T* skip, T* out, T* scratchA) {
    long long Kdim = (long long)p.k * p.k * p.C;
    long long M = (long long)p.B * p.OH * p.OW;
    LAUNCH(im2col_kernel<T>, grid1d(M * Kdim), dim3(kDefaultBlock), p, in, scratchA);
    CUDA_CHECK(cudaGetLastError());
    LAUNCH(gemm_naive_epilogue_kernel<T>, grid1d(M * p.K), dim3(kDefaultBlock), M, p.K, Kdim, (const T*)scratchA, w, bias, skip, out, p.epi);
    CUDA_CHECK(cudaGetLastError());
}

template void launch_conv_im2col<float>(const ConvParams&, const float*, const float*, const float*, const float*, float*, float*);
#ifdef ENABLE_FP16
template void launch_conv_im2col<half>(const ConvParams&, const half*, const half*, const half*, const half*, half*, half*);
#endif
