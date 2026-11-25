// main.cu - command line of the ResNet-18 inference engine.
//
//   engine --pack DIR --set {val|dev|debug} --out OUTDIR [--batch B] [--precision fp32|fp16]
//          [--conv direct|im2col] [--iters N] [--warmup W] [--dump] [--print-graph]
//
// Writes OUTDIR/logits.bin (N x 1000 float32), OUTDIR/top5.bin (N x 5 int32) and OUTDIR/timing.json.
// With --dump also writes OUTDIR/dump/<tensor>.bin for every manifest tensor (the set must fit in one batch).
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include "cuda_compat.h"
#include "executor.h"
#include "manifest.h"

#ifdef CPU_EMULATION
uint3 threadIdx, blockIdx;
dim3 blockDim, gridDim;
#endif

namespace {
struct Args {
    std::string pack, set = "dev", out = "out", precision = "fp32", conv = "direct";
    int batch = 32, iters = 1, warmup = 0;
    bool dump = false, print_graph = false;
};

void usage() {
    std::fprintf(stderr,
                 "usage: engine --pack DIR --set {val|dev|debug} --out OUTDIR [--batch B] [--precision fp32|fp16]\n"
                 "              [--conv direct|im2col] [--iters N] [--warmup W] [--dump] [--print-graph]\n");
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", name); usage(); std::exit(2); }
            return argv[++i];
        };
        if (k == "--pack") a.pack = need("--pack");
        else if (k == "--set") a.set = need("--set");
        else if (k == "--out") a.out = need("--out");
        else if (k == "--batch") a.batch = std::stoi(need("--batch"));
        else if (k == "--precision") a.precision = need("--precision");
        else if (k == "--conv") a.conv = need("--conv");
        else if (k == "--iters") a.iters = std::stoi(need("--iters"));
        else if (k == "--warmup") a.warmup = std::stoi(need("--warmup"));
        else if (k == "--dump") a.dump = true;
        else if (k == "--print-graph") a.print_graph = true;
        else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); usage(); std::exit(2); }
    }
    if (a.pack.empty()) { usage(); std::exit(2); }
    return a;
}

template <typename T>
int run_engine(const Manifest& m, const Args& a) {
    Engine<T> eng(m, a.pack, a.batch, parse_conv_mode(a.conv));
    eng.load_images(a.set);
    if (a.dump && eng.image_count() > a.batch) {
        std::fprintf(stderr, "--dump needs the whole set in one batch; set --batch %d or larger\n", eng.image_count());
        return 2;
    }
    TimingResult tr;
    eng.run(a.warmup, a.iters, tr);
    std::filesystem::create_directories(a.out);
    eng.write_outputs(a.out, tr);
    if (a.dump) {
        std::filesystem::create_directories(a.out + "/dump");
        eng.dump(a.out + "/dump");
    }
    std::printf("%s | %s | conv=%s | set=%s (%d images) | batch %d | %d warm-up + %d timed | median %.2f ms | %.1f images/s%s\n",
                tr.device.c_str(), tr.precision.c_str(), tr.conv_mode.c_str(), a.set.c_str(), tr.images, tr.batch, tr.warmup,
                tr.iters, tr.median_ms, tr.images_per_s, tr.emulation ? " [CPU EMULATION: timing is meaningless]" : "");
    std::printf("wrote %s/logits.bin, %s/top5.bin, %s/timing.json%s\n", a.out.c_str(), a.out.c_str(), a.out.c_str(),
                a.dump ? ", dump/*.bin" : "");
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    try {
        Manifest m = load_manifest(a.pack + "/manifest.json");
        if (a.print_graph) print_graph(m);
        if (a.precision == "fp32") return run_engine<float>(m, a);
        if (a.precision == "fp16") {
#ifdef ENABLE_FP16
            return run_engine<half>(m, a);
#else
            std::fprintf(stderr, "fp16 is not compiled in: configure with -DENABLE_FP16=ON (Tier 4)\n");
            return 2;
#endif
        }
        std::fprintf(stderr, "unknown precision %s\n", a.precision.c_str());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
