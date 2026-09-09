#!/usr/bin/env python3
"""Reproduce nearest-grid experiments using deterministic synthetic RGBA input.

Example:
  python3 bench/nearest_grid.py --before /path/to/baseline --after target/release/weft
All output hashes must agree. No external Python modules or ffmpeg required.
"""
import argparse
import hashlib
import json
import math
import pathlib
import platform
import random
import shlex
import statistics
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--before', type=pathlib.Path, required=True)
parser.add_argument('--after', type=pathlib.Path, required=True)
parser.add_argument('--data', type=pathlib.Path, default=pathlib.Path('bench/data/nearest-grid'))
parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('bench/out/nearest-grid.json'))
parser.add_argument('--threads', type=int, default=18)
parser.add_argument('--runs', type=int, default=15)
parser.add_argument('--args', default='', help='Extra encoder arguments, shell quoted')
parser.add_argument('--long', action='store_true', help='Also test repeated gradient/noisy clips')
args = parser.parse_args()
if args.runs < 1 or args.threads < 1:
    parser.error('runs and threads must be positive')
root = args.data
root.mkdir(parents=True, exist_ok=True)
w, h, n = 480, 270, 72
names = ['sparse', 'mixed', 'dense', 'gradient', 'cropped', 'alpha', 'noisy']
for name in names:
    if (root / (name + '.rgba')).exists():
        continue
    with (root / (name + '.rgba')).open('wb') as f:
        base = bytearray(b' @`\xff' * (w * h))
        for t in range(n):
            a = base.copy()
            if name in ('gradient', 'dense', 'mixed', 'noisy'):
                for y in range(h):
                    row = bytearray()
                    for x in range(w):
                        if name == 'noisy':
                            v = x * 1664525 + y * 1013904223 + t * 1234567 & 4294967295
                            v ^= v >> 13
                            v = v * 2246822519 & 4294967295
                            v ^= v >> 16
                            c = ((x // 3 + (v & 15)) % 256, (y // 2 + (v >> 8 & 15)) % 256, ((x + y) // 4 + (v >> 16 & 15)) % 256)
                        elif name == 'gradient':
                            c = ((x + t) // 3 % 256, (y + t) // 2 % 256, (x + y) // 4 % 256)
                        else:
                            v = (x // 8 * 17 + y // 8 * 37 + (t if name == 'dense' or (x // 24 + y // 24) % 2 else 0) * 13) % 251
                            c = (v, v * 3 % 256, v * 7 % 256)
                        row.extend((*c, 255))
                    a[y * w * 4:(y + 1) * w * 4] = row
            else:
                for y in range(40, 120):
                    x = 20 + t * 3
                    a[(y * w + x) * 4:(y * w + x + 40) * 4] = bytes((230, 140, 30, 255)) * 40
                if name == 'sparse':
                    for y in [0, h - 1]:
                        a[(y * w + t) * 4:(y * w + t + 1) * 4] = b'\xff\xff\xff\xff'
                if name == 'alpha':
                    a[3::4] = bytes((0 if i % 17 == t % 17 else 255 for i in range(w * h)))
            f.write(a)
    print('generated', name, flush=True)

if args.long:
    for name in ['gradient', 'noisy']:
        path = root / (name + '-long.rgba')
        if not path.exists():
            data = (root / (name + '.rgba')).read_bytes()
            with path.open('wb') as f:
                for _ in range(6):
                    f.write(data)
        names.append(name + '-long')
binaries = {'before': args.before.resolve(), 'after': args.after.resolve()}
opts = shlex.split(args.args)
records, hashes = [], {}
rng = random.Random(143)

def run(label, name):
    with (root / (name + '.rgba')).open('rb') as src:
        start = time.perf_counter()
        proc = subprocess.run([str(binaries[label]), '--size', f'{w}x{h}', '--format', 'rgba',
                               '--threads', str(args.threads), '--stats', *opts],
                              stdin=src, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        wall = time.perf_counter() - start
    digest = hashlib.sha256(proc.stdout).hexdigest()
    if hashes.setdefault(name, digest) != digest:
        raise RuntimeError(f'GIF mismatch: {name}, {label}')
    return dict(wall=wall, sha=digest, size=len(proc.stdout), stats=proc.stderr.decode())

for name in names:
    for label in binaries:
        run(label, name)
jobs = [(name, i) for name in names for i in range(args.runs)]
rng.shuffle(jobs)
for name, i in jobs:
    labels = list(binaries)
    rng.shuffle(labels)
    for label in labels:
        records.append(dict(name=name, label=label, iteration=i, **run(label, name)))
result = dict(platform=platform.platform(), threads=args.threads, runs=args.runs, opts=opts,
              binaries={label: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        for label, path in binaries.items()},
              inputs={name: dict(width=w, height=h, bytes=(root/(name+'.rgba')).stat().st_size,
                                  sha256=hashlib.sha256((root/(name+'.rgba')).read_bytes()).hexdigest())
                      for name in names}, records=records)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + '\n')
ratios=[]
for name in names:
    med = {label: statistics.median(r['wall'] for r in records if r['name']==name and r['label']==label)
           for label in binaries}
    ratios.append(med['before']/med['after'])
    print(f"{name}: {med['before']*1000:.2f} -> {med['after']*1000:.2f} ms; throughput {(ratios[-1]-1)*100:+.1f}%")
print(f'Geometric mean throughput gain: {(math.prod(ratios)**(1/len(ratios))-1)*100:+.1f}%')
