// fc_topk.cu - the 512 -> 1000 fully connected layer (a GEMM with M = batch) and top-5 selection.
// Naive: one thread per output logit (fc) and one thread per image doing five linear scans (topk).
// Top-k contract: class indices in descending logit order; on ties the lower class index comes first.
// Logits are fp32 in every build: the fc epilogue stores float even when T is half, so the reduced-precision
// category never rounds the final scores (the gate compares logits with the fp32 reference).
#include "kernels.h"

template <typename T>
__global__ void fc_kernel(int B, int K, int N, const T* __restrict__ in, const T* __restrict__ W,
                          const T* __restrict__ bias, float* __restrict__ out) {
    long long n = (long long)B * N;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    int col = (int)(i % N), b = (int)(i / N);
    float acc = to_f32(bias[col]);
    const T* x = in + (long long)b * K;
    const T* w = W + (long long)col * K;
    for (int k = 0; k < K; ++k) acc += to_f32(x[k]) * to_f32(w[k]);
    out[i] = acc;
}

__global__ void topk_kernel(int B, int N, int k, const float* __restrict__ logits, int* __restrict__ out) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const float* row = logits + (long long)b * N;
    for (int r = 0; r < k; ++r) {
        float best = -INFINITY;
        int bi = -1;
        for (int j = 0; j < N; ++j) {
            bool taken = false;
            for (int q = 0; q < r; ++q) taken |= (out[(long long)b * k + q] == j);
            if (taken) continue;
            float v = row[j];
            if (v > best) { best = v; bi = j; }   // strict '>' with ascending j: lowest index wins ties
        }
        out[(long long)b * k + r] = bi;
    }
}

template <typename T>
void launch_fc(int B, int K, int N, const T* in, const T* W, const T* bias, float* out) {
    LAUNCH(fc_kernel<T>, grid1d((long long)B * N), dim3(kDefaultBlock), B, K, N, in, W, bias, out);
    CUDA_CHECK(cudaGetLastError());
}
void launch_topk(int B, int N, int k, const float* logits, int* out) {
    LAUNCH(topk_kernel, grid1d(B, 64), dim3(64), B, N, k, logits, out);
    CUDA_CHECK(cudaGetLastError());
}

template void launch_fc<float>(int, int, int, const float*, const float*, const float*, float*);
#ifdef ENABLE_FP16
template void launch_fc<half>(int, int, int, const half*, const half*, const half*, float*);
#endif
