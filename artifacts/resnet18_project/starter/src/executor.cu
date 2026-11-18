// executor.cu - Engine<T> implementation. Host code only; every kernel is reached through the launch_* functions
// declared in kernels.h, so replacing a kernel never requires touching this file. The places you might touch it:
//   * forward_batch(): to fuse ops (e.g. stem conv + maxpool) or to add a new ConvMode
//   * alloc_tensors(): to re-plan buffers (ping-pong instead of one buffer per tensor)
#include "executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "kernels.h"
#include "nlohmann/json.hpp"

namespace {

std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    std::vector<char> buf((size_t)f.tellg());
    f.seekg(0);
    f.read(buf.data(), (std::streamsize)buf.size());
    return buf;
}

std::string join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    char last = a.back();
    return (last == '/' || last == '\\') ? a + b : a + "/" + b;
}

void write_file(const std::string& path, const void* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write((const char*)data, (std::streamsize)bytes);
}

template <typename T> const char* precision_name();
template <> const char* precision_name<float>() { return "fp32"; }
#ifdef ENABLE_FP16
template <> const char* precision_name<half>() { return "fp16"; }
#endif

Epi parse_epi(const std::string& e) {
    if (e == "none") return Epi::None;
    if (e == "relu") return Epi::Relu;
    return Epi::AddRelu;
}

}  // namespace

ConvMode parse_conv_mode(const std::string& s) {
    if (s == "direct") return ConvMode::Direct;
    if (s == "im2col") return ConvMode::Im2col;
    throw std::runtime_error("unknown --conv mode: " + s + " (direct | im2col)");
}

// ------------------------------------------------------------------------------------------------ construction
template <typename T>
Engine<T>::Engine(const Manifest& m, const std::string& pack_dir, int batch, ConvMode mode)
    : m_(m), pack_dir_(pack_dir), batch_(batch), mode_(mode), precision_(precision_name<T>()) {
    if (batch_ < 1) throw std::runtime_error("batch must be >= 1");
    img_bytes_ = (size_t)m_.image_shape[0] * m_.image_shape[1] * m_.image_shape[2];
    upload_weights();
    alloc_tensors();
}

template <typename T>
Engine<T>::~Engine() {
    for (auto& b : bufs_) if (b.d && b.name != "image_u8") cudaFree(b.d);
    if (d_weights_) cudaFree(d_weights_);
    if (d_images_) cudaFree(d_images_);
    if (d_scratch_) cudaFree(d_scratch_);
}

template <typename T>
void Engine<T>::upload_weights() {
    const std::string& file = (precision_ == "fp16") ? m_.weights_fp16 : m_.weights_fp32;
    std::vector<char> blob = read_file(join(pack_dir_, file));
    if (blob.size() != m_.weight_elems * sizeof(T))
        throw std::runtime_error("weights file size does not match manifest.weights.elements");
    CUDA_CHECK(cudaMalloc(&d_weights_, blob.size()));
    CUDA_CHECK(cudaMemcpy(d_weights_, blob.data(), blob.size(), cudaMemcpyHostToDevice));
}

template <typename T>
void Engine<T>::alloc_tensors() {
    bufs_.resize(m_.tensors.size());
    for (size_t i = 0; i < m_.tensors.size(); ++i) {
        const auto& t = m_.tensors[i];
        Buf& b = bufs_[i];
        b.name = t.name;
        b.elems_per_img = t.elems;
        b.is_int = (t.name == "top5");
        b.is_f32 = (t.name == "logits");
        if (t.name == "image_u8") continue;                    // points into d_images_ per batch
        b.capacity_imgs = (size_t)batch_;                      // logits/top5 are re-sized in load_images()
        size_t bytes = b.elems_per_img * b.capacity_imgs * b.elem_bytes(sizeof(T));
        CUDA_CHECK(cudaMalloc(&b.d, bytes));
    }
    // im2col scratch: the largest A over all convs at this batch size
    scratch_elems_ = 0;
    if (mode_ == ConvMode::Im2col) {
        for (const auto& op : m_.ops) {
            if (op.type != OpType::Conv) continue;
            ConvParams p{batch_, op.in_hw[0], op.in_hw[1], op.cin, op.out_hw[0], op.out_hw[1], op.cout, op.k, op.stride, op.pad, Epi::None};
            scratch_elems_ = std::max(scratch_elems_, im2col_scratch_elems(p));
        }
        CUDA_CHECK(cudaMalloc(&d_scratch_, scratch_elems_ * sizeof(T)));
    }
}

template <typename T>
void Engine<T>::load_images(const std::string& set_name) {
    auto it = m_.image_sets.find(set_name);
    if (it == m_.image_sets.end()) throw std::runtime_error("unknown image set " + set_name);
    const ImageSet& s = it->second;
    std::vector<char> blob = read_file(join(pack_dir_, s.images));
    if (blob.size() != (size_t)s.count * img_bytes_) throw std::runtime_error("image file size mismatch for set " + set_name);
    n_images_ = s.count;
    if (d_images_) cudaFree(d_images_);
    CUDA_CHECK(cudaMalloc(&d_images_, blob.size()));
    CUDA_CHECK(cudaMemcpy(d_images_, blob.data(), blob.size(), cudaMemcpyHostToDevice));
    // logits and top5 hold the whole set so nothing has to leave the device inside the timed region
    for (const char* name : {"logits", "top5"}) {
        Buf& b = bufs_[m_.tensor_index.at(name)];
        size_t need = std::max((size_t)n_images_, (size_t)batch_);
        if (need != b.capacity_imgs) {
            cudaFree(b.d);
            b.capacity_imgs = need;
            CUDA_CHECK(cudaMalloc(&b.d, b.elems_per_img * need * b.elem_bytes(sizeof(T))));
        }
    }
}

// ------------------------------------------------------------------------------------------------ pointers
template <typename T>
T* Engine<T>::tensor_ptr(const std::string& name, int b0) const {
    (void)b0;
    const Buf& b = bufs_[m_.tensor_index.at(name)];
    if (b.is_f32 || b.is_int) throw std::runtime_error(name + " is not a storage-type tensor; use logits_ptr/top5_ptr");
    return (T*)b.d;                                                        // per-batch buffer
}
template <typename T>
float* Engine<T>::logits_ptr(int b0) const {
    const Buf& b = bufs_[m_.tensor_index.at("logits")];
    return (float*)b.d + (size_t)b0 * b.elems_per_img;                     // full-set buffer
}
template <typename T>
int* Engine<T>::top5_ptr(int b0) const {
    const Buf& b = bufs_[m_.tensor_index.at("top5")];
    return (int*)b.d + (size_t)b0 * b.elems_per_img;
}

// ------------------------------------------------------------------------------------------------ forward
template <typename T>
void Engine<T>::forward_batch(int b0, int Bc) {
    const uint8_t* img = d_images_ + (size_t)b0 * img_bytes_;
    for (const auto& op : m_.ops) {
        switch (op.type) {
            case OpType::Normalize: {
                NormParams np;
                for (int i = 0; i < 3; ++i) { np.mean[i] = op.mean[i]; np.std[i] = op.std[i]; }
                launch_normalize<T>(img, tensor_ptr(op.out, b0), Bc, m_.image_shape[0], m_.image_shape[1], m_.image_shape[2], np);
                break;
            }
            case OpType::Conv: {
                ConvParams p{Bc, op.in_hw[0], op.in_hw[1], op.cin, op.out_hw[0], op.out_hw[1], op.cout, op.k, op.stride, op.pad,
                             parse_epi(op.epilogue)};
                const T* w = d_weights_ + op.w_offset;
                const T* bias = d_weights_ + op.b_offset;
                const T* skip = op.skip.empty() ? nullptr : tensor_ptr(op.skip, b0);
                T* out = tensor_ptr(op.out, b0);
                const T* in = tensor_ptr(op.in, b0);
                switch (mode_) {
                    case ConvMode::Direct: launch_conv_direct<T>(p, in, w, bias, skip, out); break;
                    case ConvMode::Im2col: launch_conv_im2col<T>(p, in, w, bias, skip, out, d_scratch_); break;
                }
                break;
            }
            case OpType::MaxPool: {
                PoolParams p{Bc, op.in_hw[0], op.in_hw[1], op.c, op.out_hw[0], op.out_hw[1], op.k, op.stride, op.pad};
                launch_maxpool<T>(p, tensor_ptr(op.in, b0), tensor_ptr(op.out, b0));
                break;
            }
            case OpType::AvgPool:
                launch_avgpool<T>(Bc, op.in_hw[0] * op.in_hw[1], op.c, tensor_ptr(op.in, b0), tensor_ptr(op.out, b0));
                break;
            case OpType::FC:
                launch_fc<T>(Bc, op.cin, op.cout, tensor_ptr(op.in, b0), d_weights_ + op.w_offset, d_weights_ + op.b_offset,
                             logits_ptr(b0));
                break;
            case OpType::TopK:
                launch_topk(Bc, m_.num_classes, op.topk, logits_ptr(b0), top5_ptr(b0));
                break;
        }
    }
}

template <typename T>
void Engine<T>::run(int warmup, int iters, TimingResult& tr) {
    if (n_images_ == 0) throw std::runtime_error("load_images() first");
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    tr.device = prop.name;
    tr.precision = precision_;
    tr.conv_mode = (mode_ == ConvMode::Direct) ? "direct" : "im2col";
    tr.images = n_images_;
    tr.batch = batch_;
    tr.iters = iters;
    tr.warmup = warmup;
#ifdef CPU_EMULATION
    tr.emulation = true;
#endif
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    tr.iter_ms.clear();
    for (int it = 0; it < warmup + iters; ++it) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int b0 = 0; b0 < n_images_; b0 += batch_) forward_batch(b0, std::min(batch_, n_images_ - b0));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (it >= warmup) tr.iter_ms.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::vector<float> sorted = tr.iter_ms;
    std::sort(sorted.begin(), sorted.end());
    tr.median_ms = sorted.empty() ? 0.f : sorted[sorted.size() / 2];
    tr.images_per_s = tr.median_ms > 0 ? n_images_ / (tr.median_ms / 1000.0) : 0.0;
}

// ------------------------------------------------------------------------------------------------ outputs
template <typename T>
void Engine<T>::write_outputs(const std::string& outdir, const TimingResult& tr) const {
    // logits are fp32 on the device in every build
    const Buf& lb = bufs_[m_.tensor_index.at("logits")];
    std::vector<float> logits((size_t)n_images_ * lb.elems_per_img);
    CUDA_CHECK(cudaMemcpy(logits.data(), lb.d, logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
    write_file(join(outdir, "logits.bin"), logits.data(), logits.size() * sizeof(float));

    const Buf& tb = bufs_[m_.tensor_index.at("top5")];
    std::vector<int> top5((size_t)n_images_ * tb.elems_per_img);
    CUDA_CHECK(cudaMemcpy(top5.data(), tb.d, top5.size() * sizeof(int), cudaMemcpyDeviceToHost));
    write_file(join(outdir, "top5.bin"), top5.data(), top5.size() * sizeof(int));

    nlohmann::json j;
    j["device"] = tr.device;
    j["precision"] = tr.precision;
    j["conv_mode"] = tr.conv_mode;
    j["emulation"] = tr.emulation;
    j["images"] = tr.images;
    j["batch"] = tr.batch;
    j["warmup"] = tr.warmup;
    j["iters"] = tr.iters;
    j["iter_ms"] = tr.iter_ms;
    j["median_ms"] = tr.median_ms;
    j["images_per_s"] = tr.images_per_s;
    std::ofstream f(join(outdir, "timing.json"));
    f << j.dump(1) << "\n";
}

template <typename T>
void Engine<T>::dump(const std::string& outdir) const {
    if (n_images_ > batch_)
        throw std::runtime_error("--dump needs the whole set in one batch: use --batch >= " + std::to_string(n_images_));
    for (const auto& b : bufs_) {
        if (b.name == "image_u8") continue;
        size_t n = (size_t)n_images_ * b.elems_per_img;
        std::string path = join(outdir, b.name + ".bin");
        if (b.is_int) {
            std::vector<int> v(n);
            CUDA_CHECK(cudaMemcpy(v.data(), b.d, n * sizeof(int), cudaMemcpyDeviceToHost));
            write_file(path, v.data(), n * sizeof(int));
        } else if (b.is_f32) {
            std::vector<float> v(n);
            CUDA_CHECK(cudaMemcpy(v.data(), b.d, n * sizeof(float), cudaMemcpyDeviceToHost));
            write_file(path, v.data(), n * sizeof(float));
        } else {
            std::vector<T> raw(n);
            CUDA_CHECK(cudaMemcpy(raw.data(), b.d, n * sizeof(T), cudaMemcpyDeviceToHost));
            std::vector<float> v(n);
            for (size_t i = 0; i < n; ++i) v[i] = to_f32(raw[i]);
            write_file(path, v.data(), n * sizeof(float));
        }
    }
}

template class Engine<float>;
#ifdef ENABLE_FP16
template class Engine<half>;
#endif
