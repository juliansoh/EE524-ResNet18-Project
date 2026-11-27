// topk_test.cu - launch_topk against a CPU reference on rows built to break the tie rule.
// Build and run from the repository root:
//   nvcc -std=c++17 -arch=sm_75 -Iengine/include tests/topk_test.cu engine/src/kernels/fc_topk.cu -o /tmp/topk_test && /tmp/topk_test
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "kernels.h"

static std::vector<int> cpu_topk(const float* row, int N, int k) {
    std::vector<int> ix(N);
    std::iota(ix.begin(), ix.end(), 0);
    std::partial_sort(ix.begin(), ix.begin() + k, ix.end(),
                      [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
    ix.resize(k);
    return ix;
}

static int run_case(const char* name, int B, int N, int k, const std::vector<float>& h) {
    float* d_logits;
    int* d_out;
    CUDA_CHECK(cudaMalloc(&d_logits, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)B * k * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_logits, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    launch_topk(B, N, k, d_logits, d_out);
    std::vector<int> got((size_t)B * k);
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, got.size() * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_logits);
    cudaFree(d_out);
    int bad = 0;
    for (int b = 0; b < B; ++b) {
        std::vector<int> want = cpu_topk(h.data() + (size_t)b * N, N, k);
        for (int r = 0; r < k; ++r) {
            if (got[(size_t)b * k + r] == want[r]) continue;
            if (bad++ < 5) std::printf("  %s: row %d rank %d: got %d, want %d\n", name, b, r, got[(size_t)b * k + r], want[r]);
        }
    }
    std::printf("%-28s B=%-3d N=%-5d k=%d  %s\n", name, B, N, k, bad ? "FAIL" : "ok");
    return bad != 0;
}

int main() {
    std::mt19937 rng(524);
    int fails = 0;
    {   // many ties: 7 distinct values over 1000 classes, so every rank is a tie spanning several lanes
        int B = 37, N = 1000;
        std::vector<float> h((size_t)B * N);
        for (auto& v : h) v = (float)(rng() % 7);
        fails += run_case("heavy ties", B, N, 5, h);
    }
    {   // all equal: the answer must be 0..4
        int B = 3, N = 1000;
        fails += run_case("all equal", B, N, 5, std::vector<float>((size_t)B * N, 1.5f));
    }
    {   // the winners sit in one lane (indices 1, 33, 65, ...) and in the tail past the last full warp
        int B = 4, N = 1000;
        std::vector<float> h((size_t)B * N, 0.f);
        for (int b = 0; b < B; ++b) {
            for (int q = 0; q < 5; ++q) h[(size_t)b * N + 1 + 32 * q] = 10.f;
            h[(size_t)b * N + N - 1] = 10.f;
        }
        fails += run_case("one lane owns all winners", B, N, 5, h);
    }
    {   // distinct values, including negatives and -inf
        int B = 64, N = 1000;
        std::uniform_real_distribution<float> u(-20.f, 20.f);
        std::vector<float> h((size_t)B * N);
        for (auto& v : h) v = u(rng);
        for (int b = 0; b < B; b += 7) h[(size_t)b * N + (b % N)] = -INFINITY;
        fails += run_case("random", B, N, 5, h);
    }
    {   // only two finite logits: -inf entries must fill ranks 3-5 in index order
        int B = 2, N = 1000;
        std::vector<float> h((size_t)B * N, -INFINITY);
        for (int b = 0; b < B; ++b) { h[(size_t)b * N + 500] = 1.f; h[(size_t)b * N + 40] = 2.f; }
        fails += run_case("-inf fills the tail", B, N, 5, h);
    }
    {   // N smaller than a warp and k < 5
        int B = 5, N = 7;
        std::vector<float> h((size_t)B * N);
        for (auto& v : h) v = (float)(rng() % 3);
        fails += run_case("short rows", B, N, 3, h);
    }
    std::printf(fails ? "FAILED\n" : "PASSED\n");
    return fails ? 1 : 0;
}
