#!/usr/bin/env python3
"""Paired weft subprocess measurements with macOS hardware counters.

Build the helper: cc -O2 bench/macos_counters.c -o /tmp/weft-counters
Each manifest item needs name, width, height, and fps; name.rgba must live
beside the manifest. Decode MP4s beforehand: decoding is not measured.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import shlex
import statistics
import subprocess
import tempfile


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', action='append', required=True, help='label=path')
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=25)
    parser.add_argument('--threads', default='18', help='comma-separated thread counts')
    parser.add_argument('--seed', type=int, default=9012026)
    parser.add_argument('--args', default='--lossy 30 --hold 12 --dither auto')
    args = parser.parse_args()
    binaries = {}
    for spec in args.binary:
        label, separator, path = spec.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+', label) or label in binaries:
            parser.error('binary labels must be unique letters/digits/hyphens/underscores')
        binaries[label] = str(Path(path).resolve(strict=True))
    threads = list(map(int, args.threads.split(',')))
    if args.runs < 1 or not threads or min(threads) < 1:
        parser.error('runs and threads must be positive')
    clips = json.loads(args.manifest.read_text())
    if not clips or len({c['name'] for c in clips}) != len(clips):
        parser.error('manifest must have at least one clip and unique names')
    for c in clips:
        raw = args.manifest.parent / (c['name'] + '.rgba')
        actual = digest(raw)
        if c.get('rgba_sha256', actual) != actual:
            parser.error(f"input checksum mismatch: {c['name']}")
        c['rgba_sha256'] = actual
    records, hashes = [], {}
    rng = random.Random(args.seed)
    helper = str(args.helper.resolve(strict=True))
    with tempfile.TemporaryDirectory(prefix='weft-counter-bench-') as scratch:
        def run(clip, count, label, iteration):
            output = Path(scratch) / (label + '.gif')
            raw = args.manifest.parent / (clip['name'] + '.rgba')
            with raw.open('rb') as source, output.open('wb') as dest:
                result = subprocess.run([
                    helper, binaries[label], '--format', 'rgba', '--size',
                    f"{clip['width']}x{clip['height']}", '--fps',
                    str(Fraction(str(clip['fps']))), '--threads', str(count),
                    '--stats', *shlex.split(args.args),
                ], stdin=source, stdout=dest, stderr=subprocess.PIPE)
            if result.returncode:
                raise RuntimeError(result.stderr.decode())
            stats, counts = result.stderr.decode().rsplit('COUNTERS ', 1)
            counter = json.loads(counts)
            if counter['instructions'] <= 0 or counter['cycles'] <= 0:
                raise RuntimeError('hardware counters are unavailable')
            sha = digest(output)
            if hashes.setdefault(clip['name'], sha) != sha:
                raise RuntimeError(f"GIF mismatch: {clip['name']}, {label}, {count} threads")
            if iteration >= 0:
                records.append(dict(clip=clip['name'], threads=count, binary=label,
                                    iteration=iteration, sha256=sha, stats=stats, **counter))

        for clip in clips:
            for count in threads:
                for label in binaries:
                    run(clip, count, label, -1)
        print('Warmups complete', flush=True)
        order = [(c, t, i) for i in range(args.runs) for c in clips for t in threads]
        rng.shuffle(order)
        for n, (clip, count, iteration) in enumerate(order, 1):
            labels = list(binaries)
            rng.shuffle(labels)
            for label in labels:
                run(clip, count, label, iteration)
            if n % len(clips) == 0:
                print(f'{n}/{len(order)} paired groups', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        binaries=binaries, binary_sha256={k: digest(Path(v)) for k, v in binaries.items()},
        helper_sha256=digest(Path(helper)), platform=platform.platform(),
        args=args.args, runs=args.runs, seed=args.seed, clips=clips, records=records,
    ), indent=2) + '\n')
    for count in threads:
        for label in binaries:
            summary = {}
            for metric in ['wall', 'cpu', 'instructions', 'cycles', 'ipc']:
                medians = [statistics.median(
                    r['instructions'] / r['cycles'] if metric == 'ipc' else r[metric]
                    for r in records if r['clip'] == c['name']
                    and r['binary'] == label and r['threads'] == count) for c in clips]
                summary[metric] = statistics.geometric_mean(medians)
            print(count, label, summary)


if __name__ == '__main__':
    main()
