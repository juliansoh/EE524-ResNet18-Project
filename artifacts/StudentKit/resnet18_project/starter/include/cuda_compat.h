// cuda_compat.h - one header that makes the same kernel source build two ways.
//
//   normal build:      real CUDA (<cuda_runtime.h>, <cuda_fp16.h>), LAUNCH(k, grid, block, args...) is k<<<grid,block>>>(args...)
//   -DCPU_EMULATION:   no GPU needed. __global__ etc. become empty, threadIdx/blockIdx are globals, and LAUNCH runs the
//                      kernel body serially for every thread of every block. Only for testing the toolchain and the
//                      checker; it is very slow and it does not emulate shared memory, __syncthreads or atomics.
//
// Students: you do not need to touch this file. Write kernels normally and launch them with LAUNCH(...) (or with
// <<<>>> directly if you do not care about the emulation build).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef CPU_EMULATION
// ------------------------------------------------------------------------------------------- real CUDA
#include <cmath>           // INFINITY, fmaxf on every host compiler
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define LAUNCH(kernel, grid, block, ...) kernel<<<(grid), (block)>>>(__VA_ARGS__)
#else
// ------------------------------------------------------------------------------------------- emulation
#include <chrono>
#include <cmath>
#include <cstring>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __launch_bounds__(...)

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct uint3 { unsigned x, y, z; };
extern uint3 threadIdx, blockIdx;
extern dim3 blockDim, gridDim;

typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
inline cudaError_t cudaMalloc(void** p, size_t n) { *p = std::malloc(n ? n : 1); return *p ? cudaSuccess : cudaErrorMemoryAllocation; }
template <typename T> inline cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc(reinterpret_cast<void**>(p), n); }
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "(emulation)"; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* n) { *n = 1; return cudaSuccess; }

struct cudaEvent_st { std::chrono::steady_clock::time_point t; };
typedef cudaEvent_st* cudaEvent_t;
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = new cudaEvent_st; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e, int = 0) { e->t = std::chrono::steady_clock::now(); return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
    *ms = std::chrono::duration<float, std::milli>(b->t - a->t).count(); return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return cudaSuccess; }

struct cudaDeviceProp {
    char name[256]; int major, minor, multiProcessorCount, clockRate, memoryClockRate, memoryBusWidth; size_t totalGlobalMem;
};
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
    std::memset(p, 0, sizeof(*p)); std::snprintf(p->name, sizeof(p->name), "CPU emulation (no GPU)");
    p->major = 0; p->minor = 0; p->multiProcessorCount = 1; return cudaSuccess;
}

// minimal IEEE binary16 stand-in so half-templated code compiles in emulation (used only with ENABLE_FP16)
struct half { unsigned short x; };
inline float __half2float(half h) {
    unsigned s = (h.x >> 15) & 1u, e = (h.x >> 10) & 0x1fu, f = h.x & 0x3ffu; unsigned bits;
    if (e == 0) { if (f == 0) bits = s << 31; else { float v = std::ldexp((float)f, -24); return s ? -v : v; } }
    else if (e == 31) bits = (s << 31) | 0x7f800000u | (f << 13);
    else bits = (s << 31) | ((e + 112u) << 23) | (f << 13);
    float out; std::memcpy(&out, &bits, 4); return out;
}
inline half __float2half(float v) {
    unsigned bits; std::memcpy(&bits, &v, 4); unsigned s = (bits >> 16) & 0x8000u; int e = (int)((bits >> 23) & 0xffu) - 127 + 15;
    unsigned f = bits & 0x7fffffu; half h;
    if (((bits >> 23) & 0xffu) == 0xffu) { h.x = (unsigned short)(s | 0x7c00u | (f ? 0x200u : 0u)); return h; }
    if (e >= 31) { h.x = (unsigned short)(s | 0x7c00u); return h; }
    if (e <= 0) {
        if (e < -10) { h.x = (unsigned short)s; return h; }
        f |= 0x800000u; unsigned shift = (unsigned)(14 - e); unsigned r = f >> shift, rem = f & ((1u << shift) - 1), halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (r & 1u))) r++;
        h.x = (unsigned short)(s | r); return h;
    }
    unsigned r = f >> 13, rem = f & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) r++;
    h.x = (unsigned short)(s | ((unsigned)e << 10) | r);
    if (r & 0x400u) h.x = (unsigned short)(s | (((unsigned)e + 1u) << 10));   // rounding carried into the exponent
    return h;
}

template <typename F, typename... Args>
void emu_launch(dim3 grid, dim3 block, F kernel, Args... args) {
    gridDim = grid; blockDim = block;
    for (unsigned bz = 0; bz < grid.z; ++bz) for (unsigned by = 0; by < grid.y; ++by) for (unsigned bx = 0; bx < grid.x; ++bx) {
        blockIdx = {bx, by, bz};
        for (unsigned tz = 0; tz < block.z; ++tz) for (unsigned ty = 0; ty < block.y; ++ty) for (unsigned tx = 0; tx < block.x; ++tx) {
            threadIdx = {tx, ty, tz};
            kernel(args...);
        }
    }
}
#define LAUNCH(kernel, grid, block, ...) emu_launch((grid), (block), (kernel), __VA_ARGS__)
#endif

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                           \
        cudaError_t err__ = (call);                                                                \
        if (err__ != cudaSuccess) {                                                                \
            std::fprintf(stderr, "CUDA error %d (%s) at %s:%d in %s\n", (int)err__,                \
                         cudaGetErrorString(err__), __FILE__, __LINE__, #call);                    \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)
