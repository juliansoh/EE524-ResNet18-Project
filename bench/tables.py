#!/usr/bin/env python3
"""Markdown tables for the design spec: shape classes from manifest.json, baseline timings from op_times.json.

usage: bench/tables.py PACK_DIR RESULTS_DIR   (RESULTS_DIR as written by bench/run_baseline.sh)
"""
import json
import math
import sys
from pathlib import Path

BATCHES_GPU = [1, 8, 32, 64, 128]
BATCHES_CPU = [1, 8]
MODES = ["direct", "im2col"]


def shape_key(op):
    if op["type"] == "conv":
        return (op["k"], op["stride"], op["cin"], op["cout"], op["in_hw"][0])
    return (op["type"],)


def class_label(op):
    if op["type"] == "conv":
        ih, oh = op["in_hw"][0], op["out_hw"][0]
        hw = f"{ih}" if ih == oh else f"{ih}->{oh}"
        return f"conv {op['k']}x{op['k']}/s{op['stride']} {op['cin']}->{op['cout']} @{hw}"
    if op["type"] == "fc":
        return f"fc {op['cin']}->{op['cout']}"
    return op["type"]


def macs(op):
    """MACs per image."""
    if op["type"] == "conv":
        return op["out_hw"][0] * op["out_hw"][1] * op["cout"] * op["k"] * op["k"] * op["cin"]
    if op["type"] == "fc":
        return op["cin"] * op["cout"]
    return 0


def min_bytes(op, tensors, batch):
    """Compulsory fp32 traffic for one launch at this batch: activations in/out (+skip) once, weights once."""
    if op["type"] not in ("conv", "fc"):
        return 0
    act = tensors[op["in"]] + (tensors[op["out"]] if op["type"] == "conv" else op["cout"])
    if op.get("skip"):
        act += tensors[op["skip"]]
    return 4 * (batch * act + op["w_count"] + op["b_count"])


def load_classes(pack):
    m = json.load(open(Path(pack) / "manifest.json"))
    tensors = {name: math.prod(shape) for name, shape in m["tensors"].items()}
    classes = {}  # key -> dict, in first-seen order
    for op in m["ops"]:
        c = classes.setdefault(shape_key(op), {"label": class_label(op), "ops": [], "type": op["type"]})
        c["ops"].append(op)
    return m, tensors, classes


def load_run(path):
    p = Path(path)
    if not (p / "op_times.json").exists():
        return None
    ot = json.load(open(p / "op_times.json"))
    tj = json.load(open(p / "timing.json"))
    return {"ops": {o["name"]: o["ms"] for o in ot["ops"]}, "images": ot["images"], "timing": tj}


def class_ms_per_img(run, cls):
    return sum(run["ops"][op["name"]] for op in cls["ops"]) / run["images"]


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(x, nd=2):
    if x is None:
        return "-"
    if x >= 100:
        return f"{x:,.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.{nd}f}" if x >= 0.01 else f"{x:.4f}"


def main():
    pack, res = sys.argv[1], Path(sys.argv[2])
    m, tensors, classes = load_classes(pack)
    total_macs = sum(macs(op) for op in m["ops"])

    print("### Compute shape classes (per image, fp32)\n")
    rows = []
    for c in classes.values():
        op = c["ops"][0]
        if op["type"] not in ("conv", "fc"):
            continue
        n = len(c["ops"])
        if op["type"] == "conv":
            M, N, K = op["out_hw"][0] * op["out_hw"][1], op["cout"], op["k"] * op["k"] * op["cin"]
        else:
            M, N, K = 1, op["cout"], op["cin"]
        gflop = 2 * n * macs(op) / 1e9
        ai = [2 * b * n * macs(op) / sum(min_bytes(o, tensors, b) for o in c["ops"]) for b in (1, 32)]
        a_mb = M * K * 4 / 2**20 if op["type"] == "conv" else 0
        rows.append([c["label"], n, f"{M}x{N}x{K}", f"{gflop:.3f}", f"{100 * n * macs(op) / total_macs:.1f}",
                     f"{op['w_count'] * 4 / 2**20:.2f}", fmt(a_mb), fmt(ai[0], 1), fmt(ai[1], 1)])
    print(table(["class", "convs", "GEMM MxNxK", "GFLOP", "% FLOP", "weights MiB", "im2col A MiB",
                 "AI B=1 (FLOP/B)", "AI B=32 (FLOP/B)"], rows))
    print(f"\nTotal: {2 * total_macs / 1e9:.3f} GFLOP per image. AI uses compulsory traffic only "
          "(each activation, skip and weight read or written once).\n")

    gpu = {(mode, b): load_run(res / "gpu" / mode / f"b{b}") for mode in MODES for b in BATCHES_GPU}
    plain = {(mode, b): res / "gpu-plain" / mode / f"b{b}" / "timing.json" for mode in MODES for b in BATCHES_GPU}
    cpu = {(mode, b): load_run(res / "cpu" / mode / f"b{b}") for mode in MODES for b in BATCHES_CPU}
    any_gpu = next((r for r in gpu.values() if r), None)
    any_cpu = next((r for r in cpu.values() if r), None)

    if any_gpu:
        print(f"### GPU end to end ({any_gpu['timing']['device']}, dev set, {any_gpu['images']} images, "
              "median of 3 after 1 warm-up)\n")
        rows = []
        for mode in MODES:
            r = [mode]
            for b in BATCHES_GPU:
                p = plain[(mode, b)]
                r.append(fmt(json.load(open(p))["images_per_s"], 1) if p.exists() else "-")
            rows.append(r)
        print(table(["conv mode"] + [f"B={b} img/s" for b in BATCHES_GPU], rows))
        print("\nPlain runs without --op-times. The per-op tables below come from separate --op-times runs.\n")

        for mode in MODES:
            ref = gpu[(mode, 32)]
            print(f"### GPU per shape class, conv={mode} (ms per image)\n")
            rows = []
            for c in classes.values():
                r = [c["label"], len(c["ops"])]
                for b in BATCHES_GPU:
                    run = gpu[(mode, b)]
                    r.append(fmt(class_ms_per_img(run, c), 3) if run else "-")
                if ref:
                    ms = class_ms_per_img(ref, c)
                    tot = sum(ref["ops"].values()) / ref["images"]
                    fl = 2 * len(c["ops"]) * macs(c["ops"][0])
                    r += [f"{100 * ms / tot:.1f}", fmt(fl / (ms * 1e6), 1) if fl else "-"]
                rows.append(r)
            print(table(["class", "ops"] + [f"B={b}" for b in BATCHES_GPU] + ["% time B=32", "GFLOP/s B=32"], rows))
            print()

    if any_cpu:
        print("### CPU reference (serial CPU_EMULATION build, std::chrono, debug set, 8 images; ms per image)\n")
        rows = []
        for c in classes.values():
            r = [c["label"]]
            for mode in MODES:
                for b in BATCHES_CPU:
                    run = cpu[(mode, b)]
                    r.append(fmt(class_ms_per_img(run, c), 3) if run else "-")
            g, cr = gpu.get(("direct", 1)), cpu.get(("direct", 1))
            r.append(f"{class_ms_per_img(cr, c) / class_ms_per_img(g, c):.2f}" if g and cr else "-")
            rows.append(r)
        tot = ["total"]
        for mode in MODES:
            for b in BATCHES_CPU:
                run = cpu[(mode, b)]
                tot.append(fmt(sum(run["ops"].values()) / run["images"], 1) if run else "-")
        g, cr = gpu.get(("direct", 1)), cpu.get(("direct", 1))
        tot.append(fmt(sum(cr["ops"].values()) / sum(g["ops"].values()) * g["images"] / cr["images"], 1)
                   if g and cr else "-")
        rows.append(tot)
        print(table(["class"] + [f"{mode} B={b}" for mode in MODES for b in BATCHES_CPU]
                    + ["GPU speedup (direct, B=1)"], rows))
        print()


if __name__ == "__main__":
    main()
