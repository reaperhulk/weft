# Portable RGBA hill climb, 2026-09-08

Baseline: merged main `bc59381cb52b3ab5b64626e053cc04f883eab113`, which
already includes the [previous ARM64 work](arm64-hill-climb.md). This round
improves throughput another **2.9%** on the M1 Max, with byte-identical GIFs.
Both retained changes apply to x86-64 and ARM64. Native x86-64 performance
has not been measured; its correctness checks ran under Rosetta.

## Measurement

Same Apple M1 Max, 10 cores, macOS 26.6, Rust 1.91.1 / LLVM 21.1.2, and
ordinary release profile as the previous round. Reused the 20 cached
Frinkiac clips: 2,880 frames, 120 seconds, 480×360 or 480×270, predecoded
RGBA. No further Frinkiac requests were needed. There is no PGO or release
workflow change.

The final default comparison uses 10 workers, one warmup per binary and
clip, then **15 timed runs per binary and clip: 600 observations**.
Clip/iteration groups and binary order within each group are shuffled by
`arm64.py`. Measurements cover the complete process, cached input reads,
GIF writes, and teardown; decoding and output hashing are outside timing.
No builds, profiling, or other benchmarks ran alongside native timings.
The tables compare per-clip medians, then take their geometric mean.

| Configuration | Clips | Runs | Baseline | Final | Throughput gain |
|---|---:|---:|---:|---:|---:|
| Defaults, 10 workers | 20 | 15 | 81.78 ms | 79.49 ms | **2.87%** |
| Defaults, 1 worker | 6 | 7 | 415.86 ms | 403.46 ms | 3.07% |
| Defaults, 8 workers | 6 | 7 | 80.97 ms | 78.77 ms | 2.79% |
| `--lossy 30 --hold 12`, 10 workers | 20 | 7 | 103.17 ms | 100.57 ms | 2.59% |

The six-clip checks use clips 1, 4, 7, 11, 15, and 20. Default CPU time
fell **3.10%** by the same geometric-mean-of-medians calculation. An earlier
eleven-run, three-binary comparison measured 2.26% overall, and isolated
the bounding-box change at 0.55%. These are small improvements on an
already optimized baseline; they are not universal speedup guarantees.

| Clip | Baseline ms | Final ms | Throughput gain |
|---|---:|---:|---:|
| 01-S01E01-2000ms | 38.48 | 35.68 | +7.8% |
| 02-S02E04-2500ms | 46.81 | 45.56 | +2.7% |
| 03-S03E08-3000ms | 56.93 | 55.01 | +3.5% |
| 04-S04E12-3500ms | 66.49 | 64.74 | +2.7% |
| 05-S05E02-4000ms | 77.69 | 76.05 | +2.2% |
| 06-S06E06-4500ms | 79.59 | 77.64 | +2.5% |
| 07-S07E10-5000ms | 62.98 | 60.93 | +3.4% |
| 08-S08E14-5500ms | 111.25 | 107.57 | +3.4% |
| 09-S09E03-6000ms | 92.77 | 89.41 | +3.8% |
| 10-S10E07-6500ms | 106.81 | 104.89 | +1.8% |
| 11-S11E11-7000ms | 114.42 | 111.97 | +2.2% |
| 12-S12E15-7500ms | 127.31 | 125.14 | +1.7% |
| 13-S14E02-8000ms | 128.10 | 125.78 | +1.8% |
| 14-S16E06-8500ms | 103.81 | 101.48 | +2.3% |
| 15-S18E10-9000ms | 131.18 | 129.18 | +1.5% |
| 16-S20E14-9500ms | 117.14 | 115.93 | +1.0% |
| 17-S23E03-9900ms | 91.35 | 89.52 | +2.0% |
| 18-S27E07-2200ms | 43.22 | 41.15 | +5.0% |
| 19-S31E11-6200ms | 62.73 | 60.70 | +3.3% |
| 20-S35E15-9800ms | 82.95 | 80.67 | +2.8% |

## Retained changes

**Center nearest-color cells using their existing corner calculations.**
The old reference point was the integer RGB point `base + 2` in each
four-value channel interval. The new reference is the OkLab midpoint of
the all-low and all-high corners. Those corners are already converted by
the eight-lane `fearless_simd` kernel, so this removes three scalar cube
roots per cell and produces a tighter bounding sphere. Palette generation
and the per-pixel distance metric are unchanged.

Across the default comparison, averaging per-clip stage medians,
nearest-map construction falls from **7.74 to 6.44 ms** (16.8% less time).
Mean candidates per cell fall from **3.5275 to 2.9255** (17.1% fewer).
Quantization also improves slightly; most repeated pixel lookups already
hit the memo cache, so shorter lists mainly help misses.

The corner cube-root kernel also needs a numerical correction for black:
evaluate the Halley division before multiplying by the current estimate.
Multiplying first underflows the numerator for zero input, eventually
producing `0/0`. The old maximum reduction ignored the resulting NaN;
using that corner in a midpoint exposed it immediately as an output
mismatch. An exhaustive regression now checks every integer RGB color
against its cell sphere, using both accurate and fast color conversions
and the existing candidate-bound slack. It passes on ARM64 and x86-64.

**Stop scanning pixels that cannot expand a GIF delta rectangle.** Search
for the first and last changed rows from opposite ends. Within those
rows, search only outside the current left/right bounds, and stop when
the bounds span the full width. Duplicate-frame handling and the emitted
rectangle are unchanged. New tests compare the descriptor against all
changed pixel coordinates, covering single rows/columns, corners, full
rows/columns, identical frames, and randomized changes.

## Rejected experiments

Trials used paired comparisons against the preceding candidate, usually
seven runs on the six-clip subset. Small promising results were checked
on the whole corpus before retention.

| Experiment | Result |
|---|---|
| Borrow full-width plain rectangles; defer their copies; reuse alternate LZW output | Neutral |
| Avoid zeroing recycled histogram scatter buffers | Neutral |
| Recycle opaque RGBA reader allocations after RGB packing | Small screen gain, no additional full-corpus gain |
| Fast conversion of the existing integer cell center | Neutral in the full comparison |
| Cache nearest indices for unchanged source rows between worker jobs | 1.5% slower in screening |
| Cache unchanged 64-pixel blocks instead | 0.5% slower in screening |
| Cache longest repeated-symbol LZW phrases with a per-pixel check | 0.4% slower overall; LZW stage about 4% slower |
| Move that check to phrase boundaries | 0.8% screen gain, negligible additional full-corpus gain |
| Phrase-oriented LZW loop without the repeat cache | Neutral in the full comparison |
| Average two separately converted cell-center samples | Small gain; superseded by reusing corner conversions |

Row/block reuse was limited by scheduling: a worker often receives
nonadjacent frames. In sampled 64-pixel blocks, adjacent-frame equality
was 16–72%, but a ten-frame gap reduced it to 0.5–43%. The extra copies
and comparisons did not pay for themselves. All rejected implementation
changes were removed.

## Validation and reproduction

- `cargo fmt --check`, `git diff --check`, and release Clippy with warnings
  denied pass.
- Native `cargo test --release`: **97 passed**, one pool benchmark ignored.
- `cargo test --release --target x86_64-apple-darwin` under Rosetta:
  **100 passed**, one pool benchmark ignored. This does not establish
  native x86-64 performance or exercise unsupported AVX-512 hardware.
- Every GIF hash matches baseline across the native default, worker-count,
  and lossy/hold benchmarks.
- All 20 corpus GIFs also match x86-64 baseline under Rosetta. The
  single-run corpus check is used for correctness, not performance claims.
- **52 additional CLI comparisons** pass, 26 per architecture: no dither
  with 2/18/64 colors, auto with 7, Bayer with 33, blue noise with 9,
  Sierra2 with 255; changing alpha at 0/127/128/255 with lossy encoding
  across all modes and 1/10 workers; filtered RGBA and Y4M. Inputs are
  fed through pipes in these checks.

Frozen binaries and raw results are local, ignored artifacts under
`bench/out/arm64/round3/`. The primary result files are
`final-default.json`, `final-workers.json`, `final-production.json`, and
`corners-fixed-full.json`; each records commands, output hashes, and
binary hashes. The translated corpus check is `x86-corpus-check.json`.
Baseline and final release binary SHA-256 values:

```text
baseline d9be219b3bf3d2b23f9983ec8ee544929ea5118c60b6c88f58e110a7d831ea75
final    adb0a83b6b6845da53c1cd02de2d8ecdd5c64f897dadef5565de293cff6853ba
```

With binaries built from merged main and this change saved separately:

```sh
python3 bench/arm64.py \
  --binary baseline=bench/out/arm64/round3/baseline \
  --binary final=bench/out/arm64/round3/final \
  --runs 15 --threads 10 \
  --output bench/out/arm64/round3/final-default.json

python3 bench/arm64.py \
  --binary baseline=bench/out/arm64/round3/baseline \
  --binary final=bench/out/arm64/round3/final \
  --runs 7 --threads 10 --args='--lossy 30 --hold 12' \
  --output bench/out/arm64/round3/final-production.json
```
