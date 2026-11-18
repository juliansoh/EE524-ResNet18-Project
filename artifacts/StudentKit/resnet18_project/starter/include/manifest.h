// manifest.h - the pack's manifest.json as C++ structs. Field meanings are documented in tools/pack_schema.py
// and in the project spec; this file only mirrors them.
#pragma once
#include <cstddef>
#include <map>
#include <string>
#include <vector>

enum class OpType { Normalize, Conv, MaxPool, AvgPool, FC, TopK };

struct Op {
    std::string name;
    OpType type;
    std::string in, out;         // tensor names
    // conv / pool
    int k = 0, stride = 0, pad = 0, cin = 0, cout = 0, c = 0;
    int in_hw[2] = {0, 0}, out_hw[2] = {0, 0};
    // conv / fc parameters (element offsets into the weights blob)
    size_t w_offset = 0, w_count = 0, b_offset = 0, b_count = 0;
    std::string epilogue;        // none | relu | add_relu
    std::string skip;            // tensor added before the ReLU (add_relu), else empty
    std::string torch_name;
    // normalize
    float mean[3] = {0, 0, 0}, std[3] = {1, 1, 1};
    // topk
    int topk = 0;
};

struct ImageSet {
    std::string images, labels, ref_logits;
    int count = 0;
};

struct TensorDesc {
    std::string name;
    std::vector<int> shape;      // per image (NHWC: [H, W, C]; [C]; [k])
    size_t elems = 1;            // product of shape
};

struct Manifest {
    std::string format;
    int version = 0;
    std::string layout;
    int num_classes = 0;
    int image_shape[3] = {0, 0, 0};
    std::string weights_fp32, weights_fp16;
    size_t weight_elems = 0;
    std::map<std::string, ImageSet> image_sets;
    std::vector<TensorDesc> tensors;                 // in manifest order
    std::map<std::string, int> tensor_index;         // name -> index into tensors
    std::vector<Op> ops;                             // in execution order
    std::string source_model;
    bool synthetic = false;
};

Manifest load_manifest(const std::string& path);
void print_graph(const Manifest& m);
const char* op_type_name(OpType t);
