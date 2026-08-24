#!/usr/bin/env python3
"""Validate weft's high-thread-count scaling on a wide machine.

Builds two binaries — the current checkout ("new") and a baseline commit
("old", default: the merge-base with origin/main, i.e. the tree before this
branch) — then sweeps --threads over both on two synthetic clips:

  shapes  low distinct-color count (exact-histogram path)
  noise   true-color, ~every pixel distinct (the pathological pass-1 case)

For each (binary, clip, threads) it reports best-of-N wall time, peak RSS,
and the read+hist stage time from --stats, and verifies every run's GIF is
byte-identical across binaries and thread counts (the fix must not change
output). Exit status is non-zero on any run failure or output mismatch;
performance numbers are reported for human judgment.

Expected on a 40-core/80-thread machine: the old binary's noise-clip time
and RSS climb steeply past ~8 threads; the new binary stays flat.

Needs: python3, cargo, git. No ffmpeg, no numpy.

Env knobs: THREADS="1 2 4 8 16 40 80"  RUNS=3  FRAMES=300  SIZE=640x360
           BASELINE=<git ref>  KEEP=1 (keep the temp workdir)
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

THREADS = [int(t) for t in os.environ.get("THREADS", "1 2 4 8 16 40 80").split()]
RUNS = int(os.environ.get("RUNS", "3"))
FRAMES = int(os.environ.get("FRAMES", "300"))
W, H = (int(v) for v in os.environ.get("SIZE", "640x360").split("x"))
FPS = 30


def sh(cmd, **kw):
    kw.setdefault("check", True)
    return subprocess.run(cmd, **kw)


def log(msg):
    print(msg, flush=True)


def build(src_dir, dest):
    sh(["cargo", "build", "--release", "--quiet"], cwd=src_dir)
    shutil.copy2(os.path.join(src_dir, "target/release/weft"), dest)


def baseline_ref():
    ref = os.environ.get("BASELINE")
    if ref:
        return ref
    for cand in (["git", "merge-base", "HEAD", "origin/main"],
                 ["git", "rev-parse", "HEAD~1"]):
        p = subprocess.run(cand, cwd=ROOT, capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout.strip()
    sys.exit("cannot determine baseline commit; set BASELINE=<ref>")


def gen_clip(kind, path):
    """640x360-style raw RGBA. shapes: animated flat blocks (few thousand
    distinct colors). noise: urandom with alpha forced opaque (~every pixel
    distinct, drives the histogram into coarse-binning)."""
    with open(path, "wb") as f:
        for i in range(FRAMES):
            if kind == "noise":
                b = bytearray(os.urandom(W * H * 4))
                b[3::4] = b"\xff" * (W * H)
                f.write(b)
            else:
                rows = []
                for y in range(H):
                    g = ((y // 40 + i // 2) % 8) * 32
                    row = bytearray()
                    for xb in range(-(-W // 40)):
                        r = ((xb + i) % 8) * 32
                        row += bytes([r, g, (i * 4) % 256, 255]) * 40
                    rows.append(bytes(row[: W * 4]))
                f.write(b"".join(rows))


# Wrapper process: its RUSAGE_CHILDREN high-water mark covers exactly one
# weft run, so peak RSS is per-run rather than a max over the whole sweep.
MEASURE = """
import json, resource, subprocess, sys, time
clip, out, cmd = sys.argv[1], sys.argv[2], sys.argv[3:]
with open(clip, "rb") as fi, open(out, "wb") as fo:
    t0 = time.monotonic()
    p = subprocess.run(cmd, stdin=fi, stdout=fo, stderr=subprocess.PIPE)
    dt = time.monotonic() - t0
r = resource.getrusage(resource.RUSAGE_CHILDREN)
print(json.dumps({"t": dt, "rss_kb": r.ru_maxrss, "rc": p.returncode,
                  "stderr": p.stderr.decode(errors="replace")}))
"""


def parse_dur(s):
    m = re.match(r"([0-9.]+)(µs|ms|s)", s)
    if not m:
        return None
    v = float(m.group(1))
    return v * {"µs": 1e-6, "ms": 1e-3, "s": 1.0}[m.group(2)]


def run_one(measure, binary, clip, out, threads):
    best = None
    for _ in range(RUNS):
        p = sh([sys.executable, measure, clip, out, binary,
                "--size", f"{W}x{H}", "--fps", str(FPS),
                "--threads", str(threads), "--stats"],
               capture_output=True, text=True)
        r = json.loads(p.stdout)
        if r["rc"] != 0:
            sys.exit(f"weft failed (threads={threads}):\n{r['stderr']}")
        if best is None or r["t"] < best["t"]:
            best = r
    m = re.search(r"read\+hist (\S+)", best["stderr"])
    best["hist"] = parse_dur(m.group(1)) if m else None
    with open(out, "rb") as f:
        best["sha"] = hashlib.sha256(f.read()).hexdigest()
    return best


def main():
    ncpu = os.cpu_count()
    log(f"machine: {ncpu} logical CPUs; clips {W}x{H} x{FRAMES} frames; "
        f"best of {RUNS} runs")

    work = tempfile.mkdtemp(prefix="weft-threads-")
    try:
        ref = baseline_ref()
        log(f"building new (current checkout) and old ({ref[:12]})...")
        build(ROOT, os.path.join(work, "weft-new"))
        wt = os.path.join(work, "baseline-src")
        sh(["git", "worktree", "add", "--detach", wt, ref], cwd=ROOT,
           capture_output=True)
        try:
            build(wt, os.path.join(work, "weft-old"))
        finally:
            sh(["git", "worktree", "remove", "--force", wt], cwd=ROOT,
               capture_output=True)

        measure = os.path.join(work, "measure.py")
        with open(measure, "w") as f:
            f.write(MEASURE)

        failures = []
        for kind in ("shapes", "noise"):
            clip = os.path.join(work, f"{kind}.rgba")
            log(f"\ngenerating {kind} clip...")
            gen_clip(kind, clip)

            log(f"=== {kind} ===")
            log(f"{'threads':>7}  {'old time':>9} {'old rss':>8} {'old hist':>9}"
                f"  {'new time':>9} {'new rss':>8} {'new hist':>9}  {'speedup':>7}")
            shas = set()
            for t in THREADS:
                res = {}
                for name in ("old", "new"):
                    binary = os.path.join(work, f"weft-{name}")
                    out = os.path.join(work, "out.gif")
                    res[name] = run_one(measure, binary, clip, out, t)
                    shas.add(res[name]["sha"])
                o, n = res["old"], res["new"]
                log(f"{t:>7}  {o['t']:>8.3f}s {o['rss_kb'] // 1024:>6}MB"
                    f" {o['hist']:>8.3f}s  {n['t']:>8.3f}s"
                    f" {n['rss_kb'] // 1024:>6}MB {n['hist']:>8.3f}s"
                    f"  {o['t'] / n['t']:>6.2f}x")
            if len(shas) == 1:
                log(f"output: byte-identical across both binaries and all "
                    f"thread counts ({next(iter(shas))[:16]}...)")
            else:
                failures.append(kind)
                log(f"output: MISMATCH — {len(shas)} distinct GIFs on {kind}")

        log("\nexpected: 'new' time and rss stay ~flat past 8 threads; on the "
            "noise clip 'old' climbs in both as threads grow.")
        if failures:
            sys.exit(f"FAIL: output mismatch on: {', '.join(failures)}")
        log("PASS: all outputs byte-identical.")
    finally:
        if os.environ.get("KEEP"):
            log(f"keeping workdir: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
