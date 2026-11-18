#!/usr/bin/env python3
"""check.py - correctness checker for the ResNet-18 engine.

End-to-end (any set):
    python check.py --pack PACK --out OUTDIR --set dev
        compares OUTDIR/logits.bin and OUTDIR/top5.bin with the pack's reference logits and labels and
        applies the leaderboard gate (pack_schema.GATE).

Per layer (debug set; engine run with --dump):
    python check.py --pack PACK --out OUTDIR --set debug --layers
        compares OUTDIR/dump/<tensor>.bin with PACK/ref/<tensor>.bin for every tensor.

Run the engine for you first:
    python check.py --pack PACK --out OUTDIR --set debug --layers --engine ./build/engine [--batch 8] [--precision fp32]

Exit status 0 = PASS, 1 = FAIL, 2 = usage/IO error. --json FILE writes the full report for the grader.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack_schema as S  # noqa: E402


def load_f32(path, shape):
    a = np.fromfile(path, dtype=np.float32)
    n = int(np.prod(shape))
    if a.size != n:
        raise SystemExit(f"{path}: expected {n} float32 values, found {a.size}")
    return a.reshape(shape)


def load_i32(path, shape):
    a = np.fromfile(path, dtype=np.int32)
    n = int(np.prod(shape))
    if a.size != n:
        raise SystemExit(f"{path}: expected {n} int32 values, found {a.size}")
    return a.reshape(shape)


def topk_rows(logits: np.ndarray, k: int) -> np.ndarray:
    """Reference tie rule: descending by value, lowest class index first on ties."""
    idx = np.arange(logits.shape[1])
    out = np.empty((logits.shape[0], k), dtype=np.int32)
    for i in range(logits.shape[0]):
        out[i] = np.lexsort((idx, -logits[i]))[:k]
    return out


# ------------------------------------------------------------------------------------------------
def check_layers(pack: str, out: str, m: dict, precision: str, tol_override: float | None) -> dict:
    tol = tol_override if tol_override is not None else S.GATE["layer_tol"][precision]
    count = m["image_sets"]["debug"]["count"]
    ref_dir = os.path.join(pack, m["image_sets"]["debug"].get("ref_dir", "ref"))
    dump_dir = os.path.join(out, "dump")
    rows, failed = [], []
    for name, shape in m["tensors"].items():
        if name == "image_u8":
            continue
        full = [count] + list(shape)
        rp, dp = os.path.join(ref_dir, f"{name}.bin"), os.path.join(dump_dir, f"{name}.bin")
        if not os.path.exists(dp):
            rows.append({"tensor": name, "status": "MISSING", "note": f"{dp} not found"})
            failed.append(name)
            continue
        if name == "top5":
            r, d = load_i32(rp, full), load_i32(dp, full)
            exact = float(np.mean(np.all(r == d, axis=1)))
            top1 = float(np.mean(r[:, 0] == d[:, 0]))
            ok = exact >= S.GATE["top5_consistency_min"]
            rows.append({"tensor": name, "status": "PASS" if ok else "FAIL", "exact_rows": exact, "top1_match": top1})
        else:
            r, d = load_f32(rp, full), load_f32(dp, full)
            scale = float(np.abs(r).max()) + 1e-12
            err = float(np.abs(d - r).max())
            rel = err / scale
            nonfinite = int((~np.isfinite(d)).sum())
            ok = (rel <= tol) and nonfinite == 0
            rows.append({"tensor": name, "status": "PASS" if ok else "FAIL", "max_abs_err": err, "max_abs_ref": scale,
                         "rel_err": rel, "nonfinite": nonfinite})
        if not ok:
            failed.append(name)
    return {"tol": tol, "count": count, "rows": rows, "failed": failed, "pass": not failed}


def print_layers(rep: dict) -> None:
    print(f"\nper-layer check on the debug set ({rep['count']} images), tolerance max|err|/max|ref| <= {rep['tol']:g}")
    print(f"  {'tensor':<12} {'status':<7} {'max|err|':>11} {'max|ref|':>11} {'rel err':>10}")
    for r in rep["rows"]:
        if r["tensor"] == "top5":
            print(f"  {r['tensor']:<12} {r['status']:<7} exact rows {r.get('exact_rows', 0):.3f}  top-1 match {r.get('top1_match', 0):.3f}")
        elif r["status"] == "MISSING":
            print(f"  {r['tensor']:<12} {r['status']:<7} {r['note']}")
        else:
            flag = "" if r["nonfinite"] == 0 else f"  ({r['nonfinite']} non-finite values)"
            print(f"  {r['tensor']:<12} {r['status']:<7} {r['max_abs_err']:>11.3e} {r['max_abs_ref']:>11.3e} {r['rel_err']:>10.2e}{flag}")
    print("  RESULT:", "PASS" if rep["pass"] else f"FAIL ({', '.join(rep['failed'])})")


# ------------------------------------------------------------------------------------------------
def check_end_to_end(pack: str, out: str, m: dict, set_name: str) -> dict:
    s = m["image_sets"][set_name]
    n, c = s["count"], m["num_classes"]
    ref = load_f32(os.path.join(pack, s["ref_logits"]), (n, c))
    labels = load_i32(os.path.join(pack, s["labels"]), (n,))
    lp, tp = os.path.join(out, "logits.bin"), os.path.join(out, "top5.bin")
    if not os.path.exists(lp) or not os.path.exists(tp):
        raise SystemExit(f"{out} must contain logits.bin and top5.bin")
    logits = load_f32(lp, (n, c))
    top5 = load_i32(tp, (n, S.TOPK))

    finite = bool(np.all(np.isfinite(logits)))
    ref_top5 = topk_rows(ref, S.TOPK)
    own_top5 = topk_rows(logits, S.TOPK)
    agree = float(np.mean(logits.argmax(1) == ref.argmax(1)))
    max_abs = float(np.abs(logits - ref).max()) if finite else float("inf")
    mean_abs = float(np.abs(logits - ref).mean()) if finite else float("inf")

    def errors(t5):
        top1 = float(np.mean(t5[:, 0] != labels)) * 100
        top5e = float(np.mean([labels[i] not in t5[i] for i in range(n)])) * 100
        return top1, top5e

    e_top1, e_top5 = errors(top5)
    r_top1, r_top5 = errors(ref_top5)
    consistency = float(np.mean(np.all(own_top5 == top5, axis=1)))
    gates = {
        "finite": finite,
        "top1_agreement": agree >= S.GATE["top1_agreement_min"],
        "top1_error_delta": abs(e_top1 - r_top1) <= S.GATE["error_delta_max_pp"],
        "top5_error_delta": abs(e_top5 - r_top5) <= S.GATE["error_delta_max_pp"],
        "top5_consistency": consistency >= S.GATE["top5_consistency_min"],
    }
    rep = {
        "set": set_name, "count": n, "top1_agreement": agree, "max_abs_logit_err": max_abs, "mean_abs_logit_err": mean_abs,
        "top1_err_pct": e_top1, "top5_err_pct": e_top5, "ref_top1_err_pct": r_top1, "ref_top5_err_pct": r_top5,
        "top5_consistency": consistency, "gates": gates, "pass": all(gates.values()),
    }
    tj = os.path.join(out, "timing.json")
    if os.path.exists(tj):
        with open(tj) as f:
            rep["timing"] = json.load(f)
    return rep


def print_end_to_end(rep: dict) -> None:
    g = rep["gates"]
    mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    print(f"\nend-to-end check on set '{rep['set']}' ({rep['count']} images)")
    print(f"  top-1 agreement with reference logits : {100 * rep['top1_agreement']:.2f}%   (gate >= {100 * S.GATE['top1_agreement_min']:.1f}%)  {mark(g['top1_agreement'])}")
    print(f"  logit error vs reference               : max {rep['max_abs_logit_err']:.3e}, mean {rep['mean_abs_logit_err']:.3e}  finite: {mark(g['finite'])}")
    print(f"  top-1 error vs labels                  : {rep['top1_err_pct']:.2f}%  (reference {rep['ref_top1_err_pct']:.2f}%, gate within {S.GATE['error_delta_max_pp']} pp)  {mark(g['top1_error_delta'])}")
    print(f"  top-5 error vs labels                  : {rep['top5_err_pct']:.2f}%  (reference {rep['ref_top5_err_pct']:.2f}%)  {mark(g['top5_error_delta'])}")
    print(f"  top5.bin consistent with own logits    : {100 * rep['top5_consistency']:.2f}% of rows  {mark(g['top5_consistency'])}")
    if "timing" in rep:
        t = rep["timing"]
        emu = "  [CPU emulation: not a score]" if t.get("emulation") else ""
        print(f"  timing.json                            : {t.get('images_per_s', 0):.1f} images/s, median {t.get('median_ms', 0):.2f} ms, "
              f"batch {t.get('batch')}, {t.get('precision')}, {t.get('device')}{emu}")
    print("  RESULT:", "PASS" if rep["pass"] else "FAIL")


# ------------------------------------------------------------------------------------------------
def verify_pack(pack: str) -> None:
    """Compare every file listed in checksums.sha256 with its SHA-256; exit 2 on any mismatch."""
    import hashlib
    path = os.path.join(pack, "checksums.sha256")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; this pack was not produced by export_pack.py")
    bad = 0
    for line in open(path):
        digest, rel = line.rstrip("\n").split("  ", 1)
        p = os.path.join(pack, rel)
        h = hashlib.sha256()
        try:
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
        except OSError:
            print(f"  MISSING  {rel}"); bad += 1; continue
        if h.hexdigest() != digest:
            print(f"  MISMATCH {rel}"); bad += 1
    print(f"pack verification: {'OK' if not bad else str(bad) + ' problem(s)'} ({path})")
    if bad:
        sys.exit(2)


def run_engine(args, m) -> None:
    cmd = [args.engine, "--pack", args.pack, "--set", args.set, "--out", args.out, "--precision", args.precision,
           "--iters", str(args.iters), "--warmup", str(args.warmup)]
    if args.conv:
        cmd += ["--conv", args.conv]
    batch = args.batch
    if args.layers:
        batch = max(batch or 0, m["image_sets"][args.set]["count"])
        cmd.append("--dump")
    if batch:
        cmd += ["--batch", str(batch)]
    cmd += args.engine_args.split()
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"engine exited with status {r.returncode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", help="engine output directory (logits.bin, top5.bin, timing.json[, dump/]); "
                                  "not needed for --verify-pack --no-e2e")
    ap.add_argument("--set", default="dev", choices=list(S.IMAGE_SETS))
    ap.add_argument("--layers", action="store_true", help="per-layer comparison (debug set, needs --dump output)")
    ap.add_argument("--no-e2e", action="store_true", help="skip the end-to-end check")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--layer-tol", type=float, default=None, help="override the per-layer tolerance")
    ap.add_argument("--engine", help="path to the engine binary; if given, run it first")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--conv", default=None, help="conv mode to pass to the engine (default: the engine's own default)")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--engine-args", default="", help="extra arguments passed through to the engine, e.g. \"--conv igemm\"")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--verify-pack", action="store_true", help="verify every pack file against checksums.sha256 first")
    args = ap.parse_args()

    if args.verify_pack:
        verify_pack(args.pack)
    if args.no_e2e and not args.layers and not args.engine:
        sys.exit(0)                      # verify-only invocation
    if not args.out:
        ap.error("--out is required unless running --verify-pack --no-e2e only")
    m = S.load_manifest(args.pack)
    if args.layers and args.set != "debug":
        ap.error("--layers compares against the debug set references; use --set debug")
    if args.engine:
        run_engine(args, m)

    report = {"pack": os.path.abspath(args.pack), "out": os.path.abspath(args.out), "precision": args.precision}
    ok = True
    if args.layers:
        rep = check_layers(args.pack, args.out, m, args.precision, args.layer_tol)
        print_layers(rep)
        report["layers"] = rep
        ok &= rep["pass"]
    if not args.no_e2e:
        rep = check_end_to_end(args.pack, args.out, m, args.set)
        print_end_to_end(rep)
        report["end_to_end"] = rep
        ok &= rep["pass"]
    report["pass"] = ok
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
