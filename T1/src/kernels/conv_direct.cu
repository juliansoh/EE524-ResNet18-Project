// conv_direct.cu - direct convolution, NHWC, with the fused epilogue (bias, optional skip, optional ReLU).
// Naive (Tier 0): one thread per output element; every input is read straight from global memory.
// Weight layout: w[co][kh][kw][ci]. Output element (b, oy, ox, co) sits at ((b*OH + oy)*OW + ox)*K + co.
#include "kernels.h"

template <typename T>
__global__ void conv_direct_kernel(ConvParams p, const T* __restrict__ in, const T* __restrict__ w,
                                   const T* __restrict__ bias, const T* __restrict__ skip, T* __restrict__ out) {
    long long n = (long long)p.B * p.OH * p.OW * p.K;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    int co = (int)(i % p.K);
    long long r = i / p.K;
    int ox = (int)(r % p.OW); r /= p.OW;
    int oy = (int)(r % p.OH);
    int b = (int)(r / p.OH);

    float acc = to_f32(bias[co]);
    for (int kh = 0; kh < p.k; ++kh) {
        int iy = oy * p.s + kh - p.p;
        if (iy < 0 || iy >= p.H) continue;
        for (int kw = 0; kw < p.k; ++kw) {
            int ix = ox * p.s + kw - p.p;
            if (ix < 0 || ix >= p.W) continue;
            const T* ip = in + (((long long)b * p.H + iy) * p.W + ix) * p.C;
            const T* wp = w + (((long long)co * p.k + kh) * p.k + kw) * p.C;
            for (int ci = 0; ci < p.C; ++ci) acc += to_f32(ip[ci]) * to_f32(wp[ci]);
        }
    }
    if (p.epi == Epi::AddRelu) acc += to_f32(skip[i]);
    if (p.epi != Epi::None) acc = fmaxf(acc, 0.0f);
    out[i] = from_f32<T>(acc);
}

template <typename T>
void launch_conv_direct(const ConvParams& p, const T* in, const T* w, const T* bias, const T* skip, T* out) {
    long long n = (long long)p.B * p.OH * p.OW * p.K;
    LAUNCH(conv_direct_kernel<T>, grid1d(n), dim3(kDefaultBlock), p, in, w, bias, skip, out);
    CUDA_CHECK(cudaGetLastError());
}

template void launch_conv_direct<float>(const ConvParams&, const float*, const float*, const float*, const float*, float*);
#ifdef ENABLE_FP16
template void launch_conv_direct<half>(const ConvParams&, const half*, const half*, const half*, const half*, half*);
#endif
