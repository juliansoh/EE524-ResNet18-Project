"""pack_schema.py - the single source of truth for the ResNet-18 final-project data pack.

Everything that must agree across export_pack.py, the starter engine, check.py, the grader and the
written spec lives here: file names, the manifest schema, the ResNet-18 graph (tensor names, op
order, shapes), the leaderboard gate thresholds, and the engine's command-line / output contract.

The starter engine reads the manifest written by this module; the spec PDF is generated from the
docstrings and tables below, so the two cannot drift.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

FORMAT = "resnet18-pack"
VERSION = 1

# ----------------------------------------------------------------------------------------------
# Files in a pack directory
# ----------------------------------------------------------------------------------------------
FILES = {
    "manifest.json":        "Graph, shapes, weight offsets, image sets and references (this schema).",
    "weights_fp32.bin":     "All folded parameters as little-endian float32, concatenated in op order.",
    "weights_fp16.bin":     "The same parameters, same element order, as IEEE binary16.",
    "images_val.bin":       "Validation images: count x 224 x 224 x 3, uint8, NHWC (row-major, RGB).",
    "labels_val.bin":       "int32 ImageNet-1k class index per validation image.",
    "ref_logits_val.bin":   "Reference logits for the validation set: count x 1000 float32.",
    "images_dev.bin":       "Dev subset of the validation images (same format); fast iteration set.",
    "labels_dev.bin":       "int32 labels for the dev subset.",
    "ref_logits_dev.bin":   "Reference logits for the dev subset.",
    "dev_indices.bin":      "int32 index into the validation set for each dev image.",
    "images_debug.bin":     "Debug set (8 images in the real pack) used for per-layer checks.",
    "labels_debug.bin":     "int32 labels for the debug set.",
    "ref_logits_debug.bin": "Reference logits for the debug set.",
    "ref/<tensor>.bin":     "Per-layer reference activation for the debug set: count x [tensor shape], float32 "
                            "(top5: int32). One file per tensor named in manifest.tensors, except image_u8.",
    "checksums.sha256":     "SHA-256 of every other file (sha256sum format); check.py --verify-pack or sha256sum -c.",
}

IMAGE_SETS = ("val", "dev", "debug")
IMG_H, IMG_W, IMG_C = 224, 224, 3
NUM_CLASSES = 1000
TOPK = 5

# Imagenette's ten WordNet ids -> ImageNet-1k class index (torchvision ordering).
IMAGENETTE_WNID_TO_INDEX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

IMAGENETTE_INDEX_TO_NAME = {0: "tench", 217: "English springer", 482: "cassette player", 491: "chain saw", 497: "church",
                            566: "French horn", 569: "garbage truck", 571: "gas pump", 574: "golf ball", 701: "parachute"}

PREPROCESS = {
    "resize": 256,
    "crop": 224,
    "interpolation": "bilinear",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "note": "torchvision ResNet18_Weights.IMAGENET1K_V1 transforms: resize shorter side to 256, center-crop 224, "
            "x/255, then (x - mean) / std per channel. Images are stored as uint8 after the crop; "
            "normalization is the engine's first op.",
}

# ----------------------------------------------------------------------------------------------
# Manifest op schema: type -> ordered list of (field, type, description)
# ----------------------------------------------------------------------------------------------
COMMON_FIELDS = [
    ("name", "string", "Unique op name (also used in timing breakdowns)."),
    ("type", "string", "One of: normalize, conv, maxpool, avgpool, fc, topk."),
    ("in", "string", "Input tensor name (must exist in manifest.tensors or be image_u8)."),
    ("out", "string", "Output tensor name (must exist in manifest.tensors)."),
]
OP_FIELDS: Dict[str, List[tuple]] = {
    "normalize": [
        ("mean", "float[3]", "Per-channel mean (RGB)."),
        ("std", "float[3]", "Per-channel std (RGB)."),
    ],
    "conv": [
        ("k", "int", "Kernel size (square)."),
        ("stride", "int", "Stride (same in H and W)."),
        ("pad", "int", "Zero padding (same on all sides)."),
        ("cin", "int", "Input channels."),
        ("cout", "int", "Output channels."),
        ("in_hw", "int[2]", "Input spatial size [H, W]."),
        ("out_hw", "int[2]", "Output spatial size [H, W] = floor((in + 2 pad - k) / stride) + 1."),
        ("w_offset", "int", "Element offset of the filter tensor in the weights blob."),
        ("w_count", "int", "cout * k * k * cin elements, laid out [cout][kh][kw][cin] (row-major)."),
        ("b_offset", "int", "Element offset of the bias vector (BN folded in)."),
        ("b_count", "int", "cout."),
        ("epilogue", "string", "none | relu | add_relu : out = f(acc + bias [+ skip])."),
        ("skip", "string|null", "Tensor added before the ReLU when epilogue = add_relu, else null."),
        ("torch_name", "string", "Name of the originating torchvision module (for reference only)."),
    ],
    "maxpool": [
        ("k", "int", "Window size."), ("stride", "int", "Stride."), ("pad", "int", "Padding (-inf outside)."),
        ("c", "int", "Channels."), ("in_hw", "int[2]", "Input [H, W]."), ("out_hw", "int[2]", "Output [H, W]."),
    ],
    "avgpool": [
        ("c", "int", "Channels."), ("in_hw", "int[2]", "Input [H, W]; output is 1 x 1 x c (global average)."),
    ],
    "fc": [
        ("cin", "int", "Input features (512)."), ("cout", "int", "Output classes (1000)."),
        ("w_offset", "int", "Element offset of the weight matrix, laid out [cout][cin]."),
        ("w_count", "int", "cout * cin."), ("b_offset", "int", "Element offset of the bias."), ("b_count", "int", "cout."),
        ("torch_name", "string", "Originating module name."),
    ],
    "topk": [
        ("k", "int", "Number of top classes (5). Output is int32 class indices, descending, lowest index on ties."),
    ],
}
EPILOGUES = ("none", "relu", "add_relu")

# ----------------------------------------------------------------------------------------------
# The ResNet-18 graph (torchvision topology, BN folded). Shapes are per image, NHWC.
# ----------------------------------------------------------------------------------------------
@dataclass
class Graph:
    tensors: Dict[str, List[int]] = field(default_factory=dict)   # name -> per-image shape
    ops: List[dict] = field(default_factory=list)


def conv_out(hw: int, k: int, s: int, p: int) -> int:
    return (hw + 2 * p - k) // s + 1


def resnet18_graph() -> Graph:
    """Return the op list and tensor table for torchvision ResNet-18 with BN folded (no offsets yet)."""
    g = Graph()
    g.tensors["image_u8"] = [IMG_H, IMG_W, IMG_C]
    g.tensors["input"] = [IMG_H, IMG_W, IMG_C]
    g.ops.append({"name": "normalize", "type": "normalize", "in": "image_u8", "out": "input",
                  "mean": PREPROCESS["mean"], "std": PREPROCESS["std"]})

    def conv(name, tin, tout, k, s, p, cin, cout, hw, epi, skip=None, torch_name=""):
        ohw = conv_out(hw, k, s, p)
        g.tensors[tout] = [ohw, ohw, cout]
        g.ops.append({"name": name, "type": "conv", "in": tin, "out": tout, "k": k, "stride": s, "pad": p,
                      "cin": cin, "cout": cout, "in_hw": [hw, hw], "out_hw": [ohw, ohw],
                      "w_offset": None, "w_count": cout * k * k * cin, "b_offset": None, "b_count": cout,
                      "epilogue": epi, "skip": skip, "torch_name": torch_name})
        return ohw

    conv("conv1", "input", "stem", 7, 2, 3, 3, 64, IMG_H, "relu", torch_name="conv1")
    g.tensors["pool0"] = [56, 56, 64]
    g.ops.append({"name": "maxpool", "type": "maxpool", "in": "stem", "out": "pool0", "k": 3, "stride": 2, "pad": 1,
                  "c": 64, "in_hw": [112, 112], "out_hw": [56, 56]})
    prev, hw, cin = "pool0", 56, 64
    for s_idx, cout in enumerate([64, 128, 256, 512], start=1):
        for b in range(2):
            stride = 2 if (s_idx > 1 and b == 0) else 1
            pre, tname = f"l{s_idx}b{b}", f"layer{s_idx}.{b}"
            ohw = conv(f"{pre}_conv1", prev, f"{pre}_h", 3, stride, 1, cin, cout, hw, "relu", torch_name=f"{tname}.conv1")
            if stride != 1 or cin != cout:
                conv(f"{pre}_ds", prev, f"{pre}_ds", 1, stride, 0, cin, cout, hw, "none", torch_name=f"{tname}.downsample.0")
                skip = f"{pre}_ds"
            else:
                skip = prev
            conv(f"{pre}_conv2", f"{pre}_h", f"{pre}_out", 3, 1, 1, cout, cout, ohw, "add_relu", skip=skip,
                 torch_name=f"{tname}.conv2")
            prev, hw, cin = f"{pre}_out", ohw, cout
    g.tensors["feat"] = [1, 1, 512]
    g.ops.append({"name": "avgpool", "type": "avgpool", "in": prev, "out": "feat", "c": 512, "in_hw": [7, 7]})
    g.tensors["logits"] = [NUM_CLASSES]
    g.ops.append({"name": "fc", "type": "fc", "in": "feat", "out": "logits", "cin": 512, "cout": NUM_CLASSES,
                  "w_offset": None, "w_count": NUM_CLASSES * 512, "b_offset": None, "b_count": NUM_CLASSES,
                  "torch_name": "fc"})
    g.tensors["top5"] = [TOPK]
    g.ops.append({"name": "top5", "type": "topk", "in": "logits", "out": "top5", "k": TOPK})
    return g


def assign_offsets(g: Graph) -> int:
    """Assign weight-blob element offsets in op order; return total element count."""
    off = 0
    for op in g.ops:
        if op["type"] in ("conv", "fc"):
            op["w_offset"] = off; off += op["w_count"]
            op["b_offset"] = off; off += op["b_count"]
    return off


def tensor_elems(shape: List[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def conv_shape_classes(g: Graph) -> List[dict]:
    """Group the convs into shape classes (same k/stride/cin/cout/out_hw) for the spec's table."""
    classes: Dict[tuple, dict] = {}
    for op in g.ops:
        if op["type"] != "conv":
            continue
        key = (op["k"], op["stride"], op["cin"], op["cout"], op["out_hw"][0])
        c = classes.setdefault(key, {"k": op["k"], "stride": op["stride"], "cin": op["cin"], "cout": op["cout"],
                                     "out": op["out_hw"][0], "names": [], "M_per_image": op["out_hw"][0] * op["out_hw"][1],
                                     "N": op["cout"], "K": op["k"] * op["k"] * op["cin"],
                                     "macs": op["out_hw"][0] * op["out_hw"][1] * op["cout"] * op["k"] * op["k"] * op["cin"]})
        c["names"].append(op["name"])
    return list(classes.values())


# ----------------------------------------------------------------------------------------------
# Leaderboard gate and categories
# ----------------------------------------------------------------------------------------------
GATE = {
    "top1_agreement_min": 0.995,     # fraction of images whose argmax matches the reference logits
    "error_delta_max_pp": 0.5,       # |top-k error - reference top-k error| in percentage points, k = 1 and 5
    "layer_tol": {"fp32": 1e-3, "fp16": 2e-2},   # per-layer max|err| / max|ref| on the debug set
    "top5_consistency_min": 1.0,     # top5.bin must equal the argsort of the team's own logits (tie rule)
}
CATEGORIES = {
    "fp32": {"label": "FP32", "roofline_images_per_s": 8.1e12 / 3.63e9,
             "rule": "single precision end to end"},
    "fp16": {"label": "FP16 reduced precision", "roofline_images_per_s": 65e12 / 3.63e9,
             "rule": "fp16 operands anywhere (tensor cores or not), fp32 accumulation; logits in fp32"},
}
TIERS = {0: "Baseline", 1: "Batched GEMM path", 2: "Implicit GEMM", 3: "Register tiling", 4: "Tensor cores",
         5: "Fusion and tuning"}

T4 = {"name": "NVIDIA T4", "sm": "sm_75", "fp32_tflops": 8.1, "fp16_tc_tflops": 65, "mem_gbs": 320,
      "sms": 40, "smem_per_block_kb": 64, "l2_mb": 4, "mem_gb": 16}
GFLOP_PER_IMAGE = 3.63

# ----------------------------------------------------------------------------------------------
# Engine command-line / output contract
# ----------------------------------------------------------------------------------------------
ENGINE_CONTRACT = {
    "usage": "engine --pack DIR --set {val|dev|debug} --out OUTDIR [--batch B] [--precision fp32|fp16] "
             "[--conv direct|im2col] [--iters N] [--warmup W] [--dump] [--print-graph]",
    "outputs": {
        "logits.bin": "count x 1000 float32, image order of the set; logits (pre-softmax), always fp32.",
        "top5.bin": "count x 5 int32, descending by logit, lowest class index first on ties.",
        "timing.json": "device, precision, batch, per-iteration ms, median ms, images_per_s, emulation flag.",
        "dump/<tensor>.bin": "With --dump (set must fit in one batch): every manifest tensor except image_u8, "
                             "count x [shape], float32 (top5 int32).",
    },
    "timing": "Compute only. All images are resident on the device before the clock starts; the timed region runs "
              "from the first normalize kernel to the last top-k kernel over all batches; D2H happens after. "
              "1 warm-up iteration, then --iters timed iterations; the median is reported.",
}

FORBIDDEN_IN_SCORED_PATH = ["cublas", "cudnn", "cutlass", "cub/", "thrust", "torch", "onnxruntime", "nvinfer", "tensorrt"]
ALLOWED_IN_SCORED_PATH = ["CUDA runtime, streams, events", "cooperative groups", "warp intrinsics (shuffle, vote, match)",
                          "WMMA and mma.sync (inline PTX) tensor-core instructions", "cuda_fp16.h conversions"]


# ----------------------------------------------------------------------------------------------
# Validation and loading
# ----------------------------------------------------------------------------------------------
class ManifestError(ValueError):
    pass


def validate_manifest(m: dict, pack_dir: Optional[str] = None) -> None:
    """Raise ManifestError on any inconsistency. If pack_dir is given, also check that files exist."""
    if m.get("format") != FORMAT or m.get("version") != VERSION:
        raise ManifestError(f"format/version mismatch: {m.get('format')} v{m.get('version')}")
    tensors = m["tensors"]
    if "image_u8" not in tensors or "input" not in tensors or "logits" not in tensors or "top5" not in tensors:
        raise ManifestError("tensors must include image_u8, input, logits, top5")
    names = set()
    total = m["weights"]["elements"]
    for op in m["ops"]:
        for f, _, _ in COMMON_FIELDS:
            if f not in op:
                raise ManifestError(f"op missing field {f}: {op}")
        if op["name"] in names:
            raise ManifestError(f"duplicate op name {op['name']}")
        names.add(op["name"])
        t = op["type"]
        if t not in OP_FIELDS:
            raise ManifestError(f"unknown op type {t}")
        for f, _, _ in OP_FIELDS[t]:
            if f not in op:
                raise ManifestError(f"{op['name']}: missing field {f}")
        if op["in"] not in tensors or op["out"] not in tensors:
            raise ManifestError(f"{op['name']}: unknown tensor {op['in']} or {op['out']}")
        if t == "conv":
            ih, oh = op["in_hw"][0], op["out_hw"][0]
            if conv_out(ih, op["k"], op["stride"], op["pad"]) != oh:
                raise ManifestError(f"{op['name']}: out_hw inconsistent with in_hw/k/stride/pad")
            if tensors[op["in"]] != [ih, op["in_hw"][1], op["cin"]]:
                raise ManifestError(f"{op['name']}: input tensor shape mismatch")
            if tensors[op["out"]] != [oh, op["out_hw"][1], op["cout"]]:
                raise ManifestError(f"{op['name']}: output tensor shape mismatch")
            if op["w_count"] != op["cout"] * op["k"] * op["k"] * op["cin"] or op["b_count"] != op["cout"]:
                raise ManifestError(f"{op['name']}: weight counts inconsistent")
            if op["epilogue"] not in EPILOGUES:
                raise ManifestError(f"{op['name']}: bad epilogue {op['epilogue']}")
            if (op["epilogue"] == "add_relu") != (op["skip"] is not None):
                raise ManifestError(f"{op['name']}: skip must be set iff epilogue is add_relu")
            if op["skip"] is not None and tensors[op["skip"]] != tensors[op["out"]]:
                raise ManifestError(f"{op['name']}: skip tensor shape must match output")
        if t in ("conv", "fc"):
            for key in ("w", "b"):
                if op[f"{key}_offset"] + op[f"{key}_count"] > total:
                    raise ManifestError(f"{op['name']}: {key} range exceeds weights blob")
        if t == "topk" and tensors[op["out"]] != [op["k"]]:
            raise ManifestError("topk output shape must be [k]")
    for s in IMAGE_SETS:
        if s not in m["image_sets"]:
            raise ManifestError(f"image set {s} missing")
    if pack_dir:
        for s in IMAGE_SETS:
            for key in ("images", "labels", "ref_logits"):
                p = os.path.join(pack_dir, m["image_sets"][s][key])
                if not os.path.exists(p):
                    raise ManifestError(f"missing file {p}")
        for key in ("fp32", "fp16"):
            if not os.path.exists(os.path.join(pack_dir, m["weights"][key])):
                raise ManifestError(f"missing weights file for {key}")


def load_manifest(pack_dir: str, validate: bool = True) -> dict:
    with open(os.path.join(pack_dir, "manifest.json")) as f:
        m = json.load(f)
    if validate:
        validate_manifest(m, pack_dir)
    return m


if __name__ == "__main__":
    g = resnet18_graph()
    n = assign_offsets(g)
    print(f"{len(g.ops)} ops, {sum(1 for o in g.ops if o['type']=='conv')} convs, {n:,} parameters, "
          f"{sum(tensor_elems(s) for k, s in g.tensors.items() if k != 'image_u8'):,} activation elements/image")
    for c in conv_shape_classes(g):
        print(f"  {c['k']}x{c['k']} s{c['stride']} {c['cin']:>3}->{c['cout']:<3} out {c['out']:>3}  "
              f"M={c['M_per_image']:>6}  N={c['N']:>4}  K={c['K']:>4}  x{len(c['names'])}  {c['macs']/1e6:7.1f} MMAC")
