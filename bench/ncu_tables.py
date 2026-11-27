#!/usr/bin/env python3
"""Markdown tables from one Nsight Compute report of one forward batch (--set full, every launch of the batch).

usage: bench/ncu_tables.py REPORT.ncu-rep PACK_DIR [BATCH]   (ncu from $NCU, else PATH; BATCH defaults to 32)
Launches map to manifest ops in order; gemm_naive_epilogue_kernel belongs to the conv before it (im2col mode).
FLOPs are the exact GEMM counts from the manifest (2 per MAC), credited to the launch that does the MACs.
"""
import csv
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

NCU = os.environ.get("NCU", "ncu")
METRICS = {
    "time_ms": "gpu__time_duration.sum",
    "regs": "launch__registers_per_thread",
    "smem": "launch__shared_mem_per_block_static",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "occ_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "occ_theory": "sm__maximum_warps_per_active_cycle_pct",
    "eligible": "smsp__warps_eligible.avg.per_cycle_active",
    "ld_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "ld_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1_hit": "l1tex__t_sector_hit_rate.pct",
    "l2_hit": "lts__t_sector_hit_rate.pct",
    "dram_rd": "dram__bytes_read.sum",
    "dram_wr": "dram__bytes_write.sum",
}
STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"


def ncu(report, *args):
    return subprocess.run([NCU, "--import", report, *args], capture_output=True, text=True, check=True).stdout


def num(s):
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def to_unit(v, unit, base):
    scale = {"": 1, "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
             "ns": 1e-6, "nsecond": 1e-6, "us": 1e-3, "usecond": 1e-3, "ms": 1, "msecond": 1, "s": 1e3, "second": 1e3}
    if v is None:
        return None
    return v * scale.get(unit, 1) if base else v


def read_raw(report):
    rows = list(csv.reader(io.StringIO(ncu(report, "--page", "raw", "--csv"))))
    head, units = rows[0], rows[1]
    col = {h: i for i, h in enumerate(head)}
    launches = []
    for r in rows[2:]:
        d = {"kernel": r[col["Kernel Name"]], "grid": r[col["Grid Size"]], "block": r[col["Block Size"]],
             "device": r[col["device__attribute_display_name"]]}
        for k, name in METRICS.items():
            i = col.get(name)
            d[k] = to_unit(num(r[i]), units[i], k in ("time_ms", "dram_rd", "dram_wr")) if i is not None else None
        stalls = {h[len(STALL_PREFIX):-len(STALL_SUFFIX)]: num(r[i]) for h, i in col.items()
                  if h.startswith(STALL_PREFIX) and h.endswith(STALL_SUFFIX) and num(r[i]) is not None}
        d["stalls"] = sorted(stalls.items(), key=lambda kv: -kv[1])
        launches.append(d)
    return launches


def read_rules(report):
    """Per launch: list of (section, est_speedup or None, text), in report order."""
    out, cur, section, rule = [], None, "", None
    header = re.compile(r"^  \S.*\(\d+, \d+, \d+\)x\(\d+, \d+, \d+\), Context")
    for line in ncu(report, "--page", "details", "--print-details", "all").splitlines():
        if header.match(line):
            cur = []
            out.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"^    Section: (.*)", line)
        if m:
            section, rule = m.group(1).strip(), None
            continue
        m = re.match(r"^    (OPT|WRN)\s+(.*)", line)
        if m:
            text = m.group(2).strip()
            sp = re.match(r"Est\. (?:Local )?Speedup: ([\d.]+)%", text)
            rule = [section, float(sp.group(1)) if sp else None, "" if sp else text]
            cur.append(rule)
            continue
        m = re.match(r"^ {10}(\S.*)", line)
        if m and rule is not None:
            rule[2] = (rule[2] + " " + m.group(1).strip()).strip()
        elif not line.strip() or line.startswith("    ---") or line.startswith("    INF"):
            rule = None
    return out


def macs(op):
    if op["type"] == "conv":
        return op["out_hw"][0] * op["out_hw"][1] * op["cout"] * op["k"] * op["k"] * op["cin"]
    return op["cin"] * op["cout"] if op["type"] == "fc" else 0


def short_kernel(name):
    return re.sub(r"^void ", "", name).split("(")[0].replace("<float>", "").replace("<__half>", "")


def first_sentence(text):
    text = re.sub(r"\s*\(https?://[^)]*\)", "", text)
    m = re.match(r"(.+?\.)(\s|$)", text)
    return (m.group(1) if m else text)[:220]


def fmt(x, nd=1):
    if x is None:
        return "-"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.{nd}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    report, pack = sys.argv[1], sys.argv[2]
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    m = json.load(open(Path(pack) / "manifest.json"))
    launches, rules = read_raw(report), read_rules(report)
    if len(rules) != len(launches):
        sys.exit(f"{len(launches)} launches in raw page but {len(rules)} in details page")

    ops, oi = m["ops"], -1
    for L in launches:
        k = short_kernel(L["kernel"])
        if not k.startswith("gemm_naive"):
            oi += 1
        op = ops[oi]
        L["op"] = op["name"]
        L["flop"] = 0 if k.startswith("im2col") else 2 * batch * macs(op)

    print(f"Device: {launches[0]['device']}. Report: `{Path(report).name}`, {len(launches)} launches.\n")
    rows = []
    for i, L in enumerate(launches):
        sec_req = L["ld_sectors"] / L["ld_requests"] if L["ld_sectors"] and L["ld_requests"] else None
        dram = (L["dram_rd"] or 0) + (L["dram_wr"] or 0)
        flop = L["flop"]
        gflops = flop / (L["time_ms"] * 1e6) if flop and L["time_ms"] else None
        ai = flop / dram if flop and dram else None
        top = ", ".join(f"{k} {v:.0f}" for k, v in L["stalls"][:2])
        rows.append([i, L["op"], short_kernel(L["kernel"]), f"{L['grid']}x{L['block']}", fmt(L["regs"], 0),
                     fmt(L["time_ms"], 3), fmt(L["sm_pct"]), fmt(L["mem_pct"]), fmt(L["dram_pct"]),
                     f"{fmt(L['occ_pct'])}/{fmt(L['occ_theory'], 0)}", fmt(L["eligible"], 2), fmt(sec_req, 1),
                     fmt(L["l1_hit"]), fmt(L["l2_hit"]), fmt(gflops, 0), fmt(ai, 1), top])
    print(table(["#", "op", "kernel", "grid x block", "regs", "time ms", "SM %", "mem %", "DRAM %",
                 "occ % (ach/theo)", "eligible warps/cyc", "ld sectors/req", "L1 hit %", "L2 hit %",
                 "GFLOP/s", "DRAM AI (FLOP/B)", "top stalls (cycles/issue)"], rows))
    total = sum(L["time_ms"] for L in launches)
    print(f"\nSum of kernel times: {total:.2f} ms for one batch of {batch}. GFLOP/s and AI use the exact GEMM FLOP "
          "count; DRAM AI uses measured DRAM bytes.\n")

    # Rules: per kernel function, the launch with the longest time stands for the group.
    print("#### Nsight Compute guidance (longest launch of each kernel)\n")
    by_kernel = {}
    for i, L in enumerate(launches):
        k = short_kernel(L["kernel"])
        if k not in by_kernel or L["time_ms"] > launches[by_kernel[k]]["time_ms"]:
            by_kernel[k] = i
    for k, i in by_kernel.items():
        L = launches[i]
        share = 100 * sum(x["time_ms"] for x in launches if short_kernel(x["kernel"]) == k) / total
        print(f"**{k}** (launch #{i}, op `{L['op']}`, {L['time_ms']:.3f} ms; all {k} launches: {share:.1f}% of batch time)\n")
        rs = sorted(rules[i], key=lambda r: -(r[1] or 0))
        print(table(["est. speedup", "section", "finding"],
                    [[f"{r[1]:.1f}%" if r[1] is not None else "-", r[0], first_sentence(r[2])] for r in rs]))
        print()


if __name__ == "__main__":
    main()
