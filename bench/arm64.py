#!/usr/bin/env python3
"""Paired, shuffled complete-process RGBA benchmarks; no decoder in timings.

Example: python3 bench/arm64.py --binary before=bench/out/arm64/baseline \
    --binary after=target/release/weft --runs 7 --output bench/out/arm64/results.json
"""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import random
import resource
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", action="append", required=True, help="label=path")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", default="10", help="comma-separated worker counts")
    parser.add_argument("--clips", default="", help="comma-separated clip numbers (default all)")
    parser.add_argument("--args", default="", help="additional weft arguments, shell-quoted")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    import shlex
    binaries = {label: str(Path(path).resolve()) for label, path in
                (b.split("=", 1) for b in args.binary)}
    clips = json.loads((ROOT / "data/frinkiac/manifest.json").read_text())
    if args.clips:
        selected = {int(v) for v in args.clips.split(",")}
        clips = [c for i, c in enumerate(clips, 1) if i in selected]
    threads = [int(t) for t in args.threads.split(",")]
    if not clips or any(t < 1 for t in threads):
        parser.error("select at least one clip and positive worker counts")
    records, hashes = [], {}
    rng = random.Random(20260908)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run(clip, workers, label, iteration):
        command = [binaries[label], "--format", "rgba", "--size",
                   f"{clip['width']}x{clip['height']}", "--fps", clip["fps"],
                   "--threads", str(workers), "--stats", *shlex.split(args.args)]
        output = args.output.parent / f"{label}.gif"
        with (ROOT / "data/frinkiac" / (clip["name"] + ".rgba")).open("rb") as src, output.open("wb") as dst:
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            start = time.perf_counter()
            proc = subprocess.run(command, stdin=src, stdout=dst, stderr=subprocess.PIPE, check=True)
            wall = time.perf_counter() - start
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        key = clip["name"]
        if key in hashes and hashes[key] != digest:
            raise RuntimeError(f"Output mismatch: {key}, {label}, {workers} workers")
        hashes[key] = digest
        if iteration >= 0:
            records.append(dict(clip=key, binary=label, threads=workers, iteration=iteration,
                                wall=wall, cpu=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                                size=output.stat().st_size, sha256=digest, stats=proc.stderr.decode()))

    for clip in clips:
        for workers in threads:
            for label in binaries:
                run(clip, workers, label, -1)
    print("Warmups complete", flush=True)
    groups = [(clip, workers, i) for i in range(args.runs) for clip in clips for workers in threads]
    rng.shuffle(groups)
    for index, (clip, workers, iteration) in enumerate(groups):
        labels = list(binaries)
        rng.shuffle(labels)
        for label in labels:
            run(clip, workers, label, iteration)
        if (index + 1) % len(clips) == 0:
            print(f"{index + 1}/{len(groups)} pairs", flush=True)
    args.output.write_text(json.dumps(dict(binaries=binaries, args=args.args, runs=args.runs,
                                          platform=platform.platform(), machine=platform.machine(),
                                          binary_sha256={label: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                                         for label, path in binaries.items()},
                                          clips=clips, records=records), indent=2) + "\n")
    for workers in threads:
        medians = {}
        for label in binaries:
            medians[label] = [statistics.median(r["wall"] for r in records
                             if r["clip"] == c["name"] and r["binary"] == label and r["threads"] == workers)
                              for c in clips]
            geomean = math.exp(statistics.mean(map(math.log, medians[label])))
            print(f"{workers} workers {label}: {geomean * 1000:.2f} ms geometric mean")
        base = next(iter(binaries))
        for label in list(binaries)[1:]:
            gain = math.exp(statistics.mean(math.log(a / b) for a, b in zip(medians[base], medians[label])))
            print(f"  {base}/{label}: {gain:.4f}x ({(gain - 1) * 100:.2f}% throughput gain)")


if __name__ == "__main__":
    main()
