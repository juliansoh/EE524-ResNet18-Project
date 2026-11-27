// pool.cu - max pooling (3x3 / s2 / p1 after the stem) and global average pooling (7x7 -> 1x1), NHWC.
// Naive: one thread per output element (maxpool) and one thread per (image, channel) (avgpool).
#include "kernels.h"

template <typename T>
__global__ void maxpool_kernel(PoolParams p, const T* __restrict__ in, T* __restrict__ out) {
    long long n = (long long)p.B * p.OH * p.OW * p.C;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = (int)(i % p.C);
    long long r = i / p.C;
    int ox = (int)(r % p.OW); r /= p.OW;
    int oy = (int)(r % p.OH);
    int b = (int)(r / p.OH);
    float m = -INFINITY;
    for (int kh = 0; kh < p.k; ++kh) {
        int iy = oy * p.s + kh - p.p;
        if (iy < 0 || iy >= p.H) continue;
        for (int kw = 0; kw < p.k; ++kw) {
            int ix = ox * p.s + kw - p.p;
            if (ix < 0 || ix >= p.W) continue;
            m = fmaxf(m, to_f32(in[(((long long)b * p.H + iy) * p.W + ix) * p.C + c]));
        }
    }
    out[i] = from_f32<T>(m);
}

template <typename T>
__global__ void avgpool_kernel(int B, int HW, int C, const T* __restrict__ in, T* __restrict__ out) {
    long long n = (long long)B * C;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = (int)(i % C), b = (int)(i / C);
    float acc = 0.0f;
    for (int hw = 0; hw < HW; ++hw) acc += to_f32(in[((long long)b * HW + hw) * C + c]);
    out[i] = from_f32<T>(acc / (float)HW);
}

template <typename T>
void launch_maxpool(const PoolParams& p, const T* in, T* out) {
    long long n = (long long)p.B * p.OH * p.OW * p.C;
    LAUNCH(maxpool_kernel<T>, grid1d(n), dim3(kDefaultBlock), p, in, out);
    CUDA_CHECK(cudaGetLastError());
}
template <typename T>
void launch_avgpool(int B, int HW, int C, const T* in, T* out) {
    LAUNCH(avgpool_kernel<T>, grid1d((long long)B * C), dim3(kDefaultBlock), B, HW, C, in, out);
    CUDA_CHECK(cudaGetLastError());
}

template void launch_maxpool<float>(const PoolParams&, const float*, float*);
template void launch_avgpool<float>(int, int, int, const float*, float*);
#ifdef ENABLE_FP16
template void launch_maxpool<half>(const PoolParams&, const half*, half*);
template void launch_avgpool<half>(int, int, int, const half*, half*);
#endif
