#!/usr/bin/env python3
"""submit.py - package a team's engine source for the leaderboard grader.

    python submit.py --team "warp-speed" --members "Ada Lovelace, Grace Hopper, Seymour Cray" \
                     --category fp32 --batch 64 --tier 2 --src ./starter \
                     [--engine-args "--conv igemm"] [--notes "..."] [--dest /courses/au26/submissions | user@host:/path] [--url https://...]

What it does:
  * collects CMakeLists.txt, src/, include/, third_party/ (and any other source files) from --src,
    skipping build directories, binaries, data packs and anything over 5 MB
  * writes submission.json (team, members, category, batch, tier, engine_args, notes, file hashes, timestamp)
  * zips everything to submissions/<team>_<UTC>.zip and prints the path and size
  * optionally copies the zip to a directory, scp's it to a host, or POSTs it to a URL

The grader builds the zip with the course toolchain, so the submission must build from these files alone.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

SKIP_DIRS = {"build", "build_emu", "build_emu16", ".git", ".vs", ".idea", "__pycache__", "out", "cmake-build-debug", "cmake-build-release"}
SKIP_EXT = {".o", ".obj", ".a", ".lib", ".so", ".dll", ".exe", ".bin", ".pptx", ".pdf", ".zip", ".tar", ".gz", ".ncu-rep", ".nsys-rep", ".sqlite"}
MAX_FILE_MB = 5
REQUIRED = ["CMakeLists.txt", "src", "include"]
CATEGORIES = ("fp32", "fp16")


def slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", s.strip()).strip("-")
    if not s:
        raise SystemExit("team name must contain letters or digits")
    return s[:40]


def collect(src: str):
    files = []
    for root, dirs, names in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith("build")]
        for n in names:
            p = os.path.join(root, n)
            if os.path.splitext(n)[1].lower() in SKIP_EXT:
                continue
            if os.path.getsize(p) > MAX_FILE_MB * 1024 * 1024:
                print(f"  skipping {p} (> {MAX_FILE_MB} MB)")
                continue
            files.append(p)
    return sorted(files)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def post_multipart(url: str, zip_path: str, meta: dict) -> str:
    import urllib.request
    boundary = "----resnet18submit" + hashlib.md5(zip_path.encode()).hexdigest()[:12]
    body = bytearray()
    for k, v in meta.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    with open(zip_path, "rb") as f:
        data = f.read()
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(zip_path)}\"\r\n" \
            f"Content-Type: application/zip\r\n\r\n".encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=bytes(body), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode(errors="replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", required=True)
    ap.add_argument("--members", required=True, help="comma-separated names")
    ap.add_argument("--category", required=True, choices=CATEGORIES)
    ap.add_argument("--batch", type=int, required=True, help="batch size the grader should pass to your engine")
    ap.add_argument("--tier", type=int, required=True, help="highest tier you claim (0-5); verified in the report")
    ap.add_argument("--src", default=".", help="your engine source tree (contains CMakeLists.txt, src/, include/)")
    ap.add_argument("--engine-args", default="", help="extra arguments the grader passes to your engine, e.g. \"--conv igemm\"")
    ap.add_argument("--notes", default="")
    ap.add_argument("--outdir", default="submissions")
    ap.add_argument("--dest", help="directory, or user@host:/path (scp)")
    ap.add_argument("--url", help="HTTP endpoint accepting a multipart POST with the zip")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    for r in REQUIRED:
        if not os.path.exists(os.path.join(src, r)):
            raise SystemExit(f"{src} does not contain {r}; point --src at your engine directory")
    team = slug(args.team)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(args.outdir, exist_ok=True)
    zip_path = os.path.join(args.outdir, f"{team}_{ts}.zip")

    files = collect(src)
    meta = {
        "team": team, "members": [m.strip() for m in args.members.split(",") if m.strip()],
        "category": args.category, "batch": args.batch, "tier": args.tier, "engine_args": args.engine_args.split(),
        "notes": args.notes, "submitted_utc": ts, "files": {os.path.relpath(p, src).replace(os.sep, "/"): sha256(p) for p in files},
    }
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, os.path.relpath(p, src).replace(os.sep, "/"))
        z.writestr("submission.json", json.dumps(meta, indent=1))
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"wrote {zip_path}  ({len(files)} files, {size_mb:.2f} MB)  team={team} category={args.category} batch={args.batch} tier={args.tier}")
    if size_mb > 25:
        print("warning: submission is large; did a data pack or build directory slip in?")

    if args.dest:
        is_scp = re.match(r"^[^\\/:]+@[^\\/:]+:", args.dest) is not None     # user@host:path, never C:\dir
        if is_scp:
            subprocess.run(["scp", zip_path, args.dest], check=True)
            print(f"copied to {args.dest}")
        else:
            os.makedirs(args.dest, exist_ok=True)
            shutil.copy(zip_path, args.dest)
            print(f"copied to {os.path.join(args.dest, os.path.basename(zip_path))}")
    if args.url:
        print("server:", post_multipart(args.url, zip_path, {"team": team, "category": args.category}).strip()[:500])


if __name__ == "__main__":
    sys.exit(main())
