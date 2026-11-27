// normalize.cu - uint8 NHWC image -> normalized T, (x/255 - mean[c]) / std[c].
// Naive: one thread per element. Easy first replacement; later this fuses into the stem conv's load.
#include "kernels.h"

// A runtime index into an array inside a by-value kernel parameter makes the compiler copy the struct to local
// memory, so the channel constants are selected instead of indexed.
__device__ __forceinline__ float channel_value(const float (&v)[3], int c) {
    return c == 0 ? v[0] : (c == 1 ? v[1] : v[2]);
}

template <typename T>
__global__ void normalize_u8_kernel(const uint8_t* __restrict__ in, T* __restrict__ out, long long n, int C, NormParams np) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = (int)(i % C);
    float v = (float)in[i] * (1.0f / 255.0f);
    out[i] = from_f32<T>((v - channel_value(np.mean, c)) / channel_value(np.std, c));
}

template <typename T>
void launch_normalize(const uint8_t* in, T* out, int B, int H, int W, int C, NormParams np) {
    long long n = (long long)B * H * W * C;
    LAUNCH(normalize_u8_kernel<T>, grid1d(n), dim3(kDefaultBlock), in, out, n, C, np);
    CUDA_CHECK(cudaGetLastError());
}

template void launch_normalize<float>(const uint8_t*, float*, int, int, int, int, NormParams);
#ifdef ENABLE_FP16
template void launch_normalize<half>(const uint8_t*, half*, int, int, int, int, NormParams);
#endif
