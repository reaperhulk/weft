#!/usr/bin/env python3
"""A/B benchmark harness for weft.

Times one or two weft binaries over a clip set, best-of-N wall clock plus the
per-stage breakdown from --stats, and (with --quality) checks that a
candidate's output has not regressed in size or PSNR/SSIM against the source.

  bench/ab.py --a target/release/weft-base --b target/release/weft --set quick
  bench/ab.py --a target/release/weft --set full --json base.json
"""
import argparse, json, os, re, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

SETS = {
    # fast iteration loop: one 5 s clip per frinkiac scene + the synthetics
    # that stress different stages (gradients: dither, mandel: palette).
    "quick": ["frink/hams-5s", "frink/goo-5s", "frink/mayor-5s",
              "frink/cater-5s", "frink/cupid-5s",
              "testsrc", "gradients", "mandel"],
    # duration scaling on one scene: separates per-frame from per-clip cost.
    "dur": [f"frink/hams-{d}s" for d in (1, 3, 5, 7, 10)],
    # large frames: cache and memory-bandwidth behavior.
    "big": ["frink/hams-10s-720p", "frink/hams-10s-1080p", "big"],
    "synth": ["testsrc", "gradients", "mandel", "life", "static", "big"],
}
SETS["frink"] = sorted({f"frink/{s}-{d}s" for s in
                        ("hams", "goo", "mayor", "cater", "cupid")
                        for d in (1, 3, 5, 7, 10)})
SETS["full"] = SETS["frink"] + SETS["synth"] + SETS["big"][:2]

# Rust's Duration Debug picks its own unit (ns/us/ms/s), so parse the unit too.
DUR = r"([\d.]+)(ns|\u00b5s|ms|s)"
STAGE_RE = re.compile(
    rf"read\+hist {DUR}.*?palette\+lut {DUR}\s+quantize\+lzw {DUR}"
    rf"\s+mux\+write {DUR}\s+total {DUR}")
UNIT_MS = {"ns": 1e-6, "\u00b5s": 1e-3, "ms": 1.0, "s": 1000.0}
STAGES = ["read_hist", "palette", "quant_lzw", "mux", "total_internal"]


def clip_path(name):
    p = DATA / (name + ".y4m")
    return p if p.exists() else DATA / (name + ".rgba")


def run_once(binary, clip, out, extra):
    """One timed run. Returns (wall_seconds, stage_dict, out_bytes)."""
    with open(clip, "rb") as fin, open(out, "wb") as fout:
        t0 = time.perf_counter()
        r = subprocess.run([binary, "--stats", *extra], stdin=fin, stdout=fout,
                           stderr=subprocess.PIPE)
        wall = time.perf_counter() - t0
    if r.returncode != 0:
        sys.exit(f"{binary} failed on {clip}:\n{r.stderr.decode()}")
    m = STAGE_RE.search(r.stderr.decode().replace("\n", " "))
    if m:
        g = m.groups()
        vals = [float(g[i]) * UNIT_MS[g[i + 1]] for i in range(0, len(g), 2)]
        stages = dict(zip(STAGES, vals))
    else:
        stages = {}
    return wall, stages, os.path.getsize(out)


def bench(binary, names, runs, extra, outdir):
    res = {}
    for name in names:
        clip = clip_path(name)
        out = outdir / (name.replace("/", "_") + ".gif")
        out.parent.mkdir(parents=True, exist_ok=True)
        run_once(binary, clip, out, extra)  # warm page cache, discard
        walls, stagesets, size = [], [], 0
        for _ in range(runs):
            w, s, size = run_once(binary, clip, out, extra)
            walls.append(w)
            stagesets.append(s)
        best = min(range(len(walls)), key=lambda i: walls[i])
        res[name] = {
            "wall": walls[best],
            "wall_med": statistics.median(walls),
            "size": size,
            "stages": stagesets[best],
            "gif": str(out),
        }
    return res


def quality(name, gif):
    """PSNR/SSIM of the decoded GIF against the source clip."""
    src = clip_path(name)
    def metric(kind, extra):
        p = subprocess.run(
            ["ffmpeg", "-v", "info", "-i", str(gif), "-i", str(src),
             "-lavfi", f"[0:v]format=rgb24[a];[1:v]format=rgb24[b];[a][b]{kind}{extra}",
             "-f", "null", "-"], capture_output=True, text=True)
        return p.stderr
    out = metric("psnr", "")
    m = re.search(r"average:([\d.inf]+)", out)
    psnr = float(m.group(1)) if m and m.group(1) != "inf" else float("inf")
    out = metric("ssim", "")
    m = re.search(r"All:([\d.]+)", out)
    return psnr, (float(m.group(1)) if m else 0.0)


def fmt(v):
    return f"{v * 1000:7.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline binary")
    ap.add_argument("--b", help="candidate binary")
    ap.add_argument("--set", default="quick", choices=sorted(SETS))
    ap.add_argument("--clips", help="comma-separated clip names, overrides --set")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--extra", default="", help="extra weft args, space separated")
    ap.add_argument("--json", help="write raw results here")
    ap.add_argument("--quality", action="store_true",
                    help="also measure PSNR/SSIM (slow)")
    ap.add_argument("--stages", action="store_true", help="print stage breakdown")
    args = ap.parse_args()

    names = args.clips.split(",") if args.clips else SETS[args.set]
    extra = args.extra.split() if args.extra else []
    outdir = ROOT / "out"
    ra = bench(args.a, names, args.runs, extra, outdir / "a")
    rb = bench(args.b, names, args.runs, extra, outdir / "b") if args.b else None

    hdr = f"{'clip':<22}{'A ms':>9}"
    if rb:
        hdr += f"{'B ms':>9}{'delta':>9}{'size A':>11}{'size B':>11}{'dsize':>8}"
    else:
        hdr += f"{'size':>11}"
    if args.stages:
        hdr += "".join(f"{s:>11}" for s in STAGES[:4])
    print(hdr)
    print("-" * len(hdr))

    tot_a = tot_b = 0.0
    for n in names:
        a = ra[n]
        tot_a += a["wall"]
        line = f"{n:<22}{fmt(a['wall'])}  "
        if rb:
            b = rb[n]
            tot_b += b["wall"]
            d = (b["wall"] / a["wall"] - 1) * 100
            ds = (b["size"] / a["size"] - 1) * 100 if a["size"] else 0
            line += f"{fmt(b['wall'])}  {d:+7.2f}%{a['size']:>11}{b['size']:>11}{ds:+7.2f}%"
        else:
            line += f"{a['size']:>11}"
        if args.stages:
            src = rb[n] if rb else a
            line += "".join(f"{src['stages'].get(s, 0):>11.1f}" for s in STAGES[:4])
        print(line)
        if args.quality and rb:
            if open(a["gif"], "rb").read() == open(b["gif"], "rb").read():
                print(f"{'':<22}  output bit-identical")
            else:
                pa, sa = quality(n, a["gif"])
                pb, sb = quality(n, b["gif"])
                flag = "  <-- REGRESSION" if (pb < pa - 0.02 or sb < sa - 0.0005) else ""
                print(f"{'':<22}  psnr {pa:.3f} -> {pb:.3f}   "
                      f"ssim {sa:.5f} -> {sb:.5f}{flag}")
    print("-" * len(hdr))
    if rb:
        print(f"{'TOTAL':<22}{fmt(tot_a)}  {fmt(tot_b)}  "
              f"{(tot_b / tot_a - 1) * 100:+7.2f}%")
    else:
        print(f"{'TOTAL':<22}{fmt(tot_a)}")

    if args.json:
        json.dump({"a": ra, "b": rb}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
