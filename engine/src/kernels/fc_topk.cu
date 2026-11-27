// fc_topk.cu - the 512 -> 1000 fully connected layer (a GEMM with M = batch) and top-5 selection.
// fc is naive: one thread per output logit.
// topk: one warp per image. Each lane keeps a sorted top-k of the logits it strides over (coalesced), then k rounds
// of a warp-shuffle argmax pick the next winner, and the lane that owns it pops it. The CPU emulation build has no
// warp shuffles, so it keeps the starter's one-thread-per-image kernel.
// Top-k contract: class indices in descending logit order; on ties the lower class index comes first.
// Logits are fp32 in every build: the fc epilogue stores float even when T is half, so the reduced-precision
// category never rounds the final scores (the gate compares logits with the fp32 reference).
#include "kernels.h"

#include <climits>

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

#ifdef CPU_EMULATION
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
#else
constexpr int kMaxTopK = 5;   // per-lane list lives in registers, so its size is fixed at compile time

// (va, ia) ranks before (vb, ib): larger logit first, lower class index on ties
__device__ __forceinline__ bool ranks_before(float va, int ia, float vb, int ib) {
    return va > vb || (va == vb && ia < ib);
}

__global__ void topk_kernel(int B, int N, int k, const float* __restrict__ logits, int* __restrict__ out) {
    int b = blockIdx.x;
    int lane = threadIdx.x;
    if (b >= B) return;
    const float* row = logits + (long long)b * N;

    float val[kMaxTopK];
    int idx[kMaxTopK];
#pragma unroll
    for (int q = 0; q < kMaxTopK; ++q) { val[q] = -INFINITY; idx[q] = INT_MAX; }

    for (int j = lane; j < N; j += 32) {
        float v = row[j];
        if (!ranks_before(v, j, val[kMaxTopK - 1], idx[kMaxTopK - 1])) continue;
        val[kMaxTopK - 1] = v;
        idx[kMaxTopK - 1] = j;
#pragma unroll
        for (int q = kMaxTopK - 1; q > 0; --q) {
            if (ranks_before(val[q], idx[q], val[q - 1], idx[q - 1])) {
                float tv = val[q]; val[q] = val[q - 1]; val[q - 1] = tv;
                int ti = idx[q]; idx[q] = idx[q - 1]; idx[q - 1] = ti;
            }
        }
    }

    for (int r = 0; r < k; ++r) {
        float bv = val[0];
        int bi = idx[0];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ranks_before(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) out[(long long)b * k + r] = (bi == INT_MAX) ? -1 : bi;   // fewer than k candidates: -1, as the starter
        if (idx[0] == bi) {
#pragma unroll
            for (int q = 0; q < kMaxTopK - 1; ++q) { val[q] = val[q + 1]; idx[q] = idx[q + 1]; }
            val[kMaxTopK - 1] = -INFINITY;
            idx[kMaxTopK - 1] = INT_MAX;
        }
    }
}
#endif

template <typename T>
void launch_fc(int B, int K, int N, const T* in, const T* W, const T* bias, float* out) {
    LAUNCH(fc_kernel<T>, grid1d((long long)B * N), dim3(kDefaultBlock), B, K, N, in, W, bias, out);
    CUDA_CHECK(cudaGetLastError());
}
void launch_topk(int B, int N, int k, const float* logits, int* out) {
#ifdef CPU_EMULATION
    LAUNCH(topk_kernel, grid1d(B, 64), dim3(64), B, N, k, logits, out);
#else
    if (k > kMaxTopK) {
        std::fprintf(stderr, "topk: k = %d exceeds kMaxTopK = %d\n", k, kMaxTopK);
        std::exit(1);
    }
    LAUNCH(topk_kernel, dim3(B), dim3(32), B, N, k, logits, out);
#endif
    CUDA_CHECK(cudaGetLastError());
}

template void launch_fc<float>(int, int, int, const float*, const float*, const float*, float*);
#ifdef ENABLE_FP16
template void launch_fc<half>(int, int, int, const half*, const half*, const half*, float*);
#endif
