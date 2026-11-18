// manifest.cpp - parse manifest.json into the Manifest structs and validate the parts the engine relies on.
#include "manifest.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "nlohmann/json.hpp"
using json = nlohmann::json;

static OpType parse_type(const std::string& t) {
    if (t == "normalize") return OpType::Normalize;
    if (t == "conv") return OpType::Conv;
    if (t == "maxpool") return OpType::MaxPool;
    if (t == "avgpool") return OpType::AvgPool;
    if (t == "fc") return OpType::FC;
    if (t == "topk") return OpType::TopK;
    throw std::runtime_error("unknown op type: " + t);
}

const char* op_type_name(OpType t) {
    switch (t) {
        case OpType::Normalize: return "normalize";
        case OpType::Conv: return "conv";
        case OpType::MaxPool: return "maxpool";
        case OpType::AvgPool: return "avgpool";
        case OpType::FC: return "fc";
        case OpType::TopK: return "topk";
    }
    return "?";
}

Manifest load_manifest(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    json j;
    f >> j;

    Manifest m;
    m.format = j.at("format").get<std::string>();
    m.version = j.at("version").get<int>();
    if (m.format != "resnet18-pack" || m.version != 1)
        throw std::runtime_error("unsupported pack format " + m.format + " v" + std::to_string(m.version));
    m.layout = j.at("layout").get<std::string>();
    m.num_classes = j.at("num_classes").get<int>();
    for (int i = 0; i < 3; ++i) m.image_shape[i] = j.at("image_shape")[i].get<int>();
    m.weights_fp32 = j.at("weights").at("fp32").get<std::string>();
    m.weights_fp16 = j.at("weights").at("fp16").get<std::string>();
    m.weight_elems = j.at("weights").at("elements").get<size_t>();
    if (j.contains("source")) {
        auto& s = j["source"];
        m.source_model = s.value("model", "");
        m.synthetic = s.value("synthetic", false);
    }
    for (auto& [name, v] : j.at("image_sets").items()) {
        ImageSet s;
        s.images = v.at("images").get<std::string>();
        s.labels = v.at("labels").get<std::string>();
        s.ref_logits = v.at("ref_logits").get<std::string>();
        s.count = v.at("count").get<int>();
        m.image_sets[name] = s;
    }
    for (auto& [name, shape] : j.at("tensors").items()) {
        TensorDesc t;
        t.name = name;
        for (auto& d : shape) { t.shape.push_back(d.get<int>()); t.elems *= (size_t)d.get<int>(); }
        m.tensor_index[name] = (int)m.tensors.size();
        m.tensors.push_back(t);
    }
    for (auto& o : j.at("ops")) {
        Op op;
        op.name = o.at("name").get<std::string>();
        op.type = parse_type(o.at("type").get<std::string>());
        op.in = o.at("in").get<std::string>();
        op.out = o.at("out").get<std::string>();
        if (!m.tensor_index.count(op.in) || !m.tensor_index.count(op.out))
            throw std::runtime_error(op.name + ": references an unknown tensor");
        switch (op.type) {
            case OpType::Normalize:
                for (int i = 0; i < 3; ++i) { op.mean[i] = o.at("mean")[i].get<float>(); op.std[i] = o.at("std")[i].get<float>(); }
                break;
            case OpType::Conv:
                op.k = o.at("k"); op.stride = o.at("stride"); op.pad = o.at("pad");
                op.cin = o.at("cin"); op.cout = o.at("cout");
                op.in_hw[0] = o.at("in_hw")[0]; op.in_hw[1] = o.at("in_hw")[1];
                op.out_hw[0] = o.at("out_hw")[0]; op.out_hw[1] = o.at("out_hw")[1];
                op.w_offset = o.at("w_offset"); op.w_count = o.at("w_count");
                op.b_offset = o.at("b_offset"); op.b_count = o.at("b_count");
                op.epilogue = o.at("epilogue").get<std::string>();
                if (!o.at("skip").is_null()) op.skip = o.at("skip").get<std::string>();
                op.torch_name = o.value("torch_name", "");
                if (op.epilogue != "none" && op.epilogue != "relu" && op.epilogue != "add_relu")
                    throw std::runtime_error(op.name + ": bad epilogue " + op.epilogue);
                if ((op.epilogue == "add_relu") != !op.skip.empty())
                    throw std::runtime_error(op.name + ": skip must be present iff epilogue is add_relu");
                if (op.w_offset + op.w_count > m.weight_elems || op.b_offset + op.b_count > m.weight_elems)
                    throw std::runtime_error(op.name + ": weight range exceeds blob");
                break;
            case OpType::MaxPool:
                op.k = o.at("k"); op.stride = o.at("stride"); op.pad = o.at("pad"); op.c = o.at("c");
                op.in_hw[0] = o.at("in_hw")[0]; op.in_hw[1] = o.at("in_hw")[1];
                op.out_hw[0] = o.at("out_hw")[0]; op.out_hw[1] = o.at("out_hw")[1];
                break;
            case OpType::AvgPool:
                op.c = o.at("c"); op.in_hw[0] = o.at("in_hw")[0]; op.in_hw[1] = o.at("in_hw")[1];
                break;
            case OpType::FC:
                op.cin = o.at("cin"); op.cout = o.at("cout");
                op.w_offset = o.at("w_offset"); op.w_count = o.at("w_count");
                op.b_offset = o.at("b_offset"); op.b_count = o.at("b_count");
                op.torch_name = o.value("torch_name", "");
                break;
            case OpType::TopK:
                op.topk = o.at("k");
                break;
        }
        m.ops.push_back(op);
    }
    return m;
}

void print_graph(const Manifest& m) {
    std::cout << "pack: " << m.source_model << (m.synthetic ? " [synthetic]" : "") << "\n"
              << "layout " << m.layout << ", " << m.weight_elems << " parameters, " << m.ops.size() << " ops, "
              << m.tensors.size() << " tensors\n\n";
    std::printf("%-12s %-10s %-10s -> %-10s  %s\n", "op", "type", "in", "out", "details");
    for (const auto& op : m.ops) {
        std::string d;
        char buf[256];
        switch (op.type) {
            case OpType::Conv:
                std::snprintf(buf, sizeof(buf), "%dx%d s%d p%d  %d->%d  %dx%d->%dx%d  epi=%s%s%s  w@%zu b@%zu",
                              op.k, op.k, op.stride, op.pad, op.cin, op.cout, op.in_hw[0], op.in_hw[1], op.out_hw[0], op.out_hw[1],
                              op.epilogue.c_str(), op.skip.empty() ? "" : " skip=", op.skip.c_str(), op.w_offset, op.b_offset);
                d = buf; break;
            case OpType::MaxPool:
                std::snprintf(buf, sizeof(buf), "%dx%d s%d p%d  c=%d  %dx%d->%dx%d", op.k, op.k, op.stride, op.pad, op.c,
                              op.in_hw[0], op.in_hw[1], op.out_hw[0], op.out_hw[1]);
                d = buf; break;
            case OpType::AvgPool:
                std::snprintf(buf, sizeof(buf), "c=%d  %dx%d->1x1", op.c, op.in_hw[0], op.in_hw[1]); d = buf; break;
            case OpType::FC:
                std::snprintf(buf, sizeof(buf), "%d->%d  w@%zu b@%zu", op.cin, op.cout, op.w_offset, op.b_offset); d = buf; break;
            case OpType::TopK:
                std::snprintf(buf, sizeof(buf), "k=%d", op.topk); d = buf; break;
            case OpType::Normalize:
                std::snprintf(buf, sizeof(buf), "mean=(%.3f,%.3f,%.3f) std=(%.3f,%.3f,%.3f)", op.mean[0], op.mean[1], op.mean[2],
                              op.std[0], op.std[1], op.std[2]);
                d = buf; break;
        }
        std::printf("%-12s %-10s %-10s -> %-10s  %s\n", op.name.c_str(), op_type_name(op.type), op.in.c_str(), op.out.c_str(), d.c_str());
    }
    std::cout << "\nimage sets:";
    for (const auto& [n, s] : m.image_sets) std::cout << " " << n << "=" << s.count;
    std::cout << "\n";
}
