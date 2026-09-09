#!/usr/bin/env python3
"""Paired complete-process batches with per-output equivalence checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', action='append', required=True, help='label=path')
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--threads', type=int, default=18)
p.add_argument('--concurrency', default='1,10')
p.add_argument('--jobs', type=int, default=24)
p.add_argument('--runs', type=int, default=5)
a = p.parse_args()
limits = [int(c) for c in a.concurrency.split(',')]
if min([a.threads, a.jobs, a.runs, *limits]) < 1:
    p.error('counts must be positive')
binaries = {label: str(Path(path).resolve()) for label, path in (b.split('=', 1) for b in a.binary)}
clips = json.loads(a.manifest.read_text())
a.output.parent.mkdir(parents=True, exist_ok=True)
work = a.output.parent / 'concurrent-outputs'
work.mkdir(exist_ok=True)
records, hashes = [], {}
rng = random.Random(20260909)


def batch(label, concurrency, iteration):
    jobs = [clips[i % len(clips)] for i in range(a.jobs)]

    def encode(job):
        i, c = job
        dst = work / f'{label}-{i}.gif'
        with (a.manifest.parent / (c['name'] + '.rgba')).open('rb') as src, dst.open('wb') as out:
            result = subprocess.run([binaries[label], '--format', 'rgba', '--size',
                                     f"{c['width']}x{c['height']}", '--fps', str(Fraction(c['fps'])),
                                     '--threads', str(a.threads), '--stats'], stdin=src, stdout=out,
                                    stderr=subprocess.PIPE, check=True)
        return c['name'], dst, result.stderr.decode()

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outputs = list(pool.map(encode, enumerate(jobs)))
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    results = []
    for name, path, stats in outputs:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes.setdefault(name, digest) != digest:
            raise RuntimeError(f'GIF mismatch: {label}, {concurrency}, {name}')
        results.append(dict(clip=name, sha256=digest, stats=stats))
    if iteration >= 0:
        records.append(dict(binary=label, concurrency=concurrency, iteration=iteration,
                            wall=wall, ms_per_encode=wall * 1000 / a.jobs,
                            cpu=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                            outputs=results))


for concurrency in limits:
    for label in binaries:
        batch(label, concurrency, -1)
order = [(c, i) for c in limits for i in range(a.runs)]
rng.shuffle(order)
for concurrency, iteration in order:
    labels = list(binaries)
    rng.shuffle(labels)
    for label in labels:
        batch(label, concurrency, iteration)
    print(f'Finished concurrency {concurrency}, pair {iteration + 1}', flush=True)
a.output.write_text(json.dumps(dict(platform=platform.platform(), binaries=binaries,
                                    binary_sha256={l: hashlib.sha256(Path(b).read_bytes()).hexdigest()
                                                   for l, b in binaries.items()},
                                    clips=clips, threads=a.threads, jobs=a.jobs, runs=a.runs,
                                    records=records), indent=2) + '\n')
for concurrency in limits:
    medians = {l: statistics.median(r['ms_per_encode'] for r in records
                                   if r['binary'] == l and r['concurrency'] == concurrency)
               for l in binaries}
    print(concurrency, medians)
