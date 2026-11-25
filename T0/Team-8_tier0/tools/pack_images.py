#!/usr/bin/env python3
"""pack_images.py - pull images out of the data pack as PNGs (for slides, or to see what the debug images are).

    python pack_images.py --pack PACK --set debug --out imgs/                 # every image of the set
    python pack_images.py --pack PACK --set val --index 12 --index 3000 --out imgs/
    python pack_images.py --pack PACK --set debug --sheet debug_sheet.png     # one labelled contact sheet

Files are named <set>_<index>_<label>_<class>.png. These are the 224x224 crops exactly as the engine sees them
(uint8, before normalization). The original full-resolution JPEGs are under <imagenette>/imagenette2/val/<wnid>/.
Needs numpy and Pillow.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack_schema as S  # noqa: E402


def load_set(pack: str, m: dict, set_name: str):
    s = m["image_sets"][set_name]
    n = s["count"]
    imgs = np.fromfile(os.path.join(pack, s["images"]), dtype=np.uint8).reshape(n, S.IMG_H, S.IMG_W, S.IMG_C)
    labels = np.fromfile(os.path.join(pack, s["labels"]), dtype=np.int32)
    return imgs, labels


def main():
    from PIL import Image, ImageDraw
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--set", default="debug", choices=list(S.IMAGE_SETS))
    ap.add_argument("--index", type=int, action="append", help="image index within the set (repeatable); default: all")
    ap.add_argument("--out", help="directory for one PNG per image")
    ap.add_argument("--sheet", help="write a labelled contact sheet PNG")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()
    if not args.out and not args.sheet:
        ap.error("give --out and/or --sheet")

    m = S.load_manifest(args.pack)
    imgs, labels = load_set(args.pack, m, args.set)
    idx = args.index if args.index else list(range(imgs.shape[0]))
    name = lambda lab: S.IMAGENETTE_INDEX_TO_NAME.get(int(lab), f"class{int(lab)}")  # noqa: E731

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for i in idx:
            fn = f"{args.set}_{i:04d}_{int(labels[i])}_{name(labels[i]).replace(' ', '-')}.png"
            Image.fromarray(imgs[i]).save(os.path.join(args.out, fn))
        print(f"wrote {len(idx)} PNG(s) to {args.out}")

    if args.sheet:
        cols = max(1, min(args.cols, len(idx)))
        rows = (len(idx) + cols - 1) // cols
        cell, pad, cap = S.IMG_W, 8, 22
        sheet = Image.new("RGB", (cols * (cell + pad) + pad, rows * (cell + cap + pad) + pad), (245, 244, 234))
        draw = ImageDraw.Draw(sheet)
        for k, i in enumerate(idx):
            x = pad + (k % cols) * (cell + pad)
            y = pad + (k // cols) * (cell + cap + pad)
            sheet.paste(Image.fromarray(imgs[i]), (x, y))
            draw.text((x + 2, y + cell + 4), f"#{i}  {name(labels[i])} ({int(labels[i])})", fill=(38, 38, 38))
        sheet.save(args.sheet)
        print(f"wrote {args.sheet} ({len(idx)} images, {cols} columns)")


if __name__ == "__main__":
    main()
