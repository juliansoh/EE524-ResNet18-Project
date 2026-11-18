// executor.h - the engine: device buffers, the manifest walk, batching, timing and output files.
//
// Engine<T> is templated on the storage type (float, or half when built with ENABLE_FP16). Arithmetic inside
// kernels is fp32 either way; logits are written as fp32 regardless of T.
#pragma once
#include <string>
#include <vector>

#include "cuda_compat.h"
#include "manifest.h"

enum class ConvMode { Direct, Im2col };
ConvMode parse_conv_mode(const std::string& s);

struct TimingResult {
    std::vector<float> iter_ms;          // one entry per timed iteration (all batches)
    float median_ms = 0.f;
    double images_per_s = 0.0;
    int images = 0, batch = 0, iters = 0, warmup = 0;
    std::string device, precision, conv_mode;
    bool emulation = false;
};

template <typename T>
class Engine {
public:
    Engine(const Manifest& m, const std::string& pack_dir, int batch, ConvMode mode);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // upload every image of the named set (val | dev | debug) to the device
    void load_images(const std::string& set_name);
    // warm-up + timed iterations; each iteration runs the whole set through the network (compute only)
    void run(int warmup, int iters, TimingResult& tr);
    // logits.bin (fp32), top5.bin (int32), timing.json
    void write_outputs(const std::string& outdir, const TimingResult& tr) const;
    // dump/<tensor>.bin for every manifest tensor except image_u8 (requires the set to fit in one batch)
    void dump(const std::string& outdir) const;

    int image_count() const { return n_images_; }

private:
    struct Buf {
        std::string name;
        size_t elems_per_img = 0;    // from the manifest tensor shape
        size_t capacity_imgs = 0;    // batch, or the whole set for logits/top5
        bool is_int = false;         // top5
        bool is_f32 = false;         // logits: fp32 in every build
        void* d = nullptr;
        size_t elem_bytes(size_t sizeof_T) const { return is_int ? sizeof(int) : (is_f32 ? sizeof(float) : sizeof_T); }
    };

    const Manifest& m_;
    std::string pack_dir_;
    int batch_;
    ConvMode mode_;
    std::string precision_;

    T* d_weights_ = nullptr;
    uint8_t* d_images_ = nullptr;     // whole image set, uint8 NHWC
    int n_images_ = 0;
    size_t img_bytes_ = 0;
    T* d_scratch_ = nullptr;          // im2col scratch
    size_t scratch_elems_ = 0;
    std::vector<Buf> bufs_;           // parallel to m_.tensors

    T* tensor_ptr(const std::string& name, int b0) const;    // pointer for a batch starting at image b0 (storage type T)
    float* logits_ptr(int b0) const;                         // fp32 logits, full-set buffer
    int* top5_ptr(int b0) const;
    void forward_batch(int b0, int Bc);
    void upload_weights();
    void alloc_tensors();
};

#ifdef ENABLE_FP16
using FP16Engine = Engine<half>;
#endif
using FP32Engine = Engine<float>;
