ARM64 RGBA hill climb, 2026-09-08

The final source build gains **17.8% throughput**, and PGO raises
that to **23.2%**, over commit
`fa2e45da7bd97bf2c2e6fbdbc6ff7cd74cc2d0e5`. All 20 clips improve and all
tested GIF hashes match. The tested PGO binary is installed at
`target/release/weft` and preserved as `bench/out/arm64/round2/final-pgo`.
The ordinary release build is `bench/out/arm64/round2/final-source`.

Measured on an Apple M1 Max, 8 performance cores + 2 efficiency cores, 64 GiB
RAM, macOS 26.6, Rust 1.91.1 / LLVM 21.1.2. Builds use the repository's release
settings: fat LTO, one codegen unit, baseline ARM64 CPU target. These results
apply to this machine, toolchain and workload.

The corpus contains 20 excerpts from [Frinkiac](https://frinkiac.com/), spanning
S01–S35 and 2.000–9.875 seconds: 2,880 frames / 120 seconds at 24 fps. Fifteen
clips are 480×360 and five are 480×270, as delivered by Frinkiac. They were
predecoded to RGBA without scaling or frame-rate conversion: 1.84 GB of raw
input. [The manifest](frinkiac-manifest.json) records episode windows, dimensions,
frame counts and SHA-256 hashes. Media are in ignored `bench/data/frinkiac/`.

The final comparison uses defaults and `--format rgba --fps 24 --threads 10
--stats`. Each clip/binary gets one warmup, followed by nine measured runs:
**900 timed observations across five binaries**. Clip/iteration groups and binary
order within each group are shuffled with a fixed seed. Timings include the
complete process, cached regular-file input, GIF writes, startup and teardown.
Decoding and output hashing are outside the timer. No builds, downloads,
profiling or other experiments run concurrently with these timings.

Figures are geometric means of per-clip median times. Throughput gain is
baseline time / candidate time minus one; CPU-time reduction uses child-process
user + system CPU time. Previous builds are measured again in this same run.

| Build | Wall time | Throughput gain | CPU-time reduction |
|---|---:|---:|---:|
| Original baseline | 96.69 ms | 0.0% | 0.0% |
| Previous source build | 91.11 ms | 6.1% | 7.1% |
| Previous source + PGO | 87.45 ms | 10.6% | 10.5% |
| Final source build | 82.07 ms | 17.8% | 19.1% |
| Final source + PGO | 78.48 ms | 23.2% | 22.9% |

PGO adds 4.6% throughput over the final
ordinary release build. Training uses only the ten odd-numbered clips, once
with defaults and once with `--lossy 30 --hold 12`. PGO gains over baseline are
**23.3% on training clips and 23.1% on the ten held-out clips**.

| Clip | Seconds | Baseline ms | Source ms | PGO ms | PGO gain |
|---|---:|---:|---:|---:|---:|
| 01 S01E01 | 2.000 | 44.88 | 38.65 | 36.06 | 24.4% |
| 02 S02E04 | 2.500 | 55.01 | 47.24 | 45.66 | 20.5% |
| 03 S03E08 | 3.000 | 66.26 | 56.99 | 54.85 | 20.8% |
| 04 S04E12 | 3.500 | 76.23 | 66.66 | 63.95 | 19.2% |
| 05 S05E02 | 4.000 | 93.01 | 78.01 | 75.39 | 23.4% |
| 06 S06E06 | 4.500 | 95.70 | 79.88 | 76.11 | 25.7% |
| 07 S07E10 | 5.000 | 75.46 | 63.23 | 61.11 | 23.5% |
| 08 S08E14 | 5.500 | 131.98 | 111.70 | 107.46 | 22.8% |
| 09 S09E03 | 6.000 | 107.14 | 93.07 | 88.13 | 21.6% |
| 10 S10E07 | 6.500 | 126.49 | 107.34 | 103.20 | 22.6% |
| 11 S11E11 | 7.000 | 135.63 | 114.93 | 110.20 | 23.1% |
| 12 S12E15 | 7.500 | 146.95 | 127.21 | 121.78 | 20.7% |
| 13 S14E02 | 8.000 | 153.17 | 129.16 | 122.61 | 24.9% |
| 14 S16E06 | 8.500 | 122.18 | 103.84 | 99.07 | 23.3% |
| 15 S18E10 | 9.000 | 160.23 | 131.67 | 124.81 | 28.4% |
| 16 S20E14 | 9.500 | 144.32 | 117.96 | 112.43 | 28.4% |
| 17 S23E03 | 9.875 | 107.35 | 91.37 | 86.95 | 23.5% |
| 18 S27E07 | 2.167 | 50.21 | 43.07 | 41.45 | 21.1% |
| 19 S31E11 | 6.167 | 71.93 | 62.81 | 60.25 | 19.4% |
| 20 S35E15 | 9.792 | 101.77 | 83.72 | 79.87 | 27.4% |

The continued pass began with macOS `sample`, using a release build with line
tables and a repeated nine-second clip to keep the process alive. Samples
identified the dependent LZW dictionary probe loop, quantization, and histogram
scanning as substantial CPU consumers. This extended, piped input was used for
finding hotspots; the final timings use the original clips and regular-file
input. Disassembly identified hash arithmetic and bounds checks in the LZW loop.

| Stage, ms | Baseline | Source | PGO |
|---|---:|---:|---:|
| read+hist | 23.59 | 21.14 | 19.69 |
| median_cut | 4.12 | 4.14 | 3.94 |
| lloyd | 3.50 | 2.25 | 2.31 |
| nearest-map | 10.05 | 7.73 | 7.34 |
| quantize | 30.09 | 26.48 | 25.44 |
| lzw | 15.29 | 11.21 | 10.87 |

Stage values are separate geometric means, so they do not sum to complete-process
wall time. LZW time falls by 26.7%
in the ordinary source build. Histogram, quantization and palette work also
contribute.

The retained changes are:

- **NEON histogram run scanning through `fearless_simd`.** Compare eight RGBA
  pixels at once, form a lane mask and find the exact first mismatch, including
  inside the final vector. A scalar equality guard avoids vector setup on
  different neighboring pixels. Transparency handling is unchanged.
- **LZW dictionary layout and hash.** Apple ARM64 uses a 128 KiB table and an
  XOR/shift hash instead of a multiply. The suffix shift follows the GIF code
  width, so small alphabets also spread across the table. A fixed-size boxed
  array proves masked probe indices are in bounds. Dictionary codes, insertion
  order and encoded bytes remain unchanged.
- **Compact nearest-color memo entries.** Apple ARM64 stores the 24-bit query
  and palette index in four bytes instead of eight, doubling entry count within
  the same byte budget. A fixed 256-entry packed palette table permits
  bounds-check-free RGB reconstruction; callers needing only the index can
  eliminate that palette load.
- **Earlier worker parking on Apple ARM64.** Keep the initial 10 µs spin, then
  park instead of repeatedly yielding to the scheduler for another 200 µs.
  The default worker count remains unchanged.
- **Earlier palette improvements.** `fearless_simd` filters candidates in
  sixteen-lane batches, keeps Lloyd minima and indices in eight lanes, packs
  three cube roots into four lanes with a normal padding lane, and uses
  four-float padded palette colors for candidate-distance loads. Floating-point
  operation order and palette-index tie breaking are preserved.
- **Earlier cache-budget and dispatch tuning.** Apple ARM64 has a 16 MiB
  total cache budget, with a 1 MiB per-worker ceiling. ARM64 uses its baseline
  SIMD level directly, avoiding the redundant `OnceLock` load.

Every new explicit SIMD operation uses `fearless_simd`. Existing x86 SIMD
kernels remain. Non-Apple targets retain their original memo-entry layout,
cache budget, LZW table size/hash and worker-yield settings. Fixed-size table
types are shared portable changes. Linux/musl and x86 binaries were not built
or timed on this host.

Several plausible changes lost in paired screens and were removed:

| Continued experiment | Result / decision |
|---|---|
| Four cache probes with SIMD tag comparisons | About 1.4% slower with either 64-bit or 32-bit memo entries |
| Quantize identical-pixel runs once and fill indices | 4.9% slower |
| Cache tags in native RGBA byte order | 1.4% slower |
| Pack opaque RGBA buffers to RGB in place | 1.9% slower |
| Cache the last LZW child per prefix | 0.5% slower |
| Four- or eight-lane dithering threshold kernels | Neutral to 0.4% slower |
| 256 KiB LZW table | Small subset gain reversed to a loss on the full corpus |
| 50 µs worker-yield window | Small gain; parking after the initial spin won the longer comparison |
| Fixed seven-bit LZW suffix shift | Severe small-palette collision regression; replaced with code-width-dependent shifting |

The fixed-shift regression was caught by `--dither none --colors 18` validation:
on three clips its LZW stage rose from about 9 ms to 44 ms. Small symbol values
restricted home slots to a small part of the table. The final adaptive shift
fixes that; the repeated 18-color comparison below covers all 20 clips.
The pre-fix binaries are retained only as experimental artifacts.

The first pass also rejected direct NEON intrinsics in favor of equivalent
`fearless_simd` code, ARM prefetch hints, 4/32 MiB cache budgets, explicit SIMD
packed-RGB expansion, retaining opaque frames as RGBA, `target-cpu=apple-m1`,
and removing packed cube roots. Some initial screens overlapped the paced
download; they selected candidates, rather than establishing final performance.
The continued pass used the completed local corpus throughout.

Three paired runs per clip/binary at one and eight workers verify the final
builds again: 360 timed observations, with matching hashes across worker counts.

| Workers | Baseline ms | Source ms | PGO ms | Source gain | PGO gain |
|---|---:|---:|---:|---:|---:|
| 1 | 538.78 | 448.78 | 437.18 | 20.1% | 23.2% |
| 8 | 98.02 | 86.79 | 82.40 | 12.9% | 19.0% |
| 10 | 96.69 | 82.07 | 78.48 | 17.8% | 23.2% |

Ten workers remain fastest here. Additional full-corpus comparisons use three
paired runs per clip/binary:

| Mode | Source gain | PGO gain |
|---|---:|---:|
| `--lossy 30 --hold 12` | 13.4% | 17.8% |
| `--dither none --colors 18` | 10.3% | 13.2% |

All 20 clips also match with `--smooth 16 --lossy 30 --hold 12`. Selected clips
match with Bayer, blue noise, Sierra2, and no-dither palettes of 2, 7, 9, 33, 64
and 129 colors. Those single-run checks establish output compatibility rather
than reliable performance differences.

`cargo test --release` passes **82 unit tests and 13 integration tests**, with
one pool microbenchmark ignored. Coverage includes all 16,777,216 sRGB colors
for cube-root equivalence, histogram vector boundaries and partial tails,
memo-cache misses/hits/collisions and the white sentinel, palette ties,
dictionary reuse across GIF alphabet widths, alpha, YUV input, prefilters and
thread-count determinism. `cargo fmt --check` and release Clippy with
`-D warnings` pass.

Peak RSS spot checks use the nine-second S18E10 clip with ten workers, three
fresh processes per binary, and macOS child `getrusage`:

| Build | Median peak MiB | Observed range MiB |
|---|---:|---:|
| baseline | 190.8 | 185.8–194.0 |
| source | 186.0 | 185.8–188.2 |
| pgo | 179.5 | 179.2–182.3 |

Observed peaks vary with allocation and scheduling; they are not memory limits.
The compact cache keeps the previous pass's byte budget unchanged. The larger
LZW table adds 96 KiB per encoder, at most 0.94 MiB across ten workers.
Compared with the original baseline, the memo-budget change also adds 5 MiB
across ten workers.

To reproduce:

```sh
python3 bench/fetch_frinkiac.py
mkdir -p bench/out/arm64
cargo build --release
cp target/release/weft bench/out/arm64/optimized-current
python3 bench/build_pgo.py --output bench/out/arm64/weft-pgo
python3 bench/arm64.py \
  --binary baseline=bench/out/arm64/baseline \
  --binary source=bench/out/arm64/optimized-current \
  --binary pgo=bench/out/arm64/weft-pgo \
  --runs 9 --threads 10 --output bench/out/arm64/repeat.json
```

In a fresh checkout, build the baseline commit separately and pass its binary
path. `build_pgo.py` needs `rustup component add llvm-tools`. It trains with
fresh counters, leaves the profile-use binary in `target/release/weft`, and
copies the executable, merged profile and training metadata to the chosen
output prefix. A normal `cargo build --release` rebuilds without PGO. The
benchmark runner checks hashes across every binary, iteration and worker count
and records wall time, CPU time, output size, checksums and stage statistics.

PGO suits CI-produced prebuilt artifacts: train an instrumented build on a
cached corpus, then publish the final profile-use binary. Users need no profiles
or training media. Train each runnable target separately and regenerate
profiles for the release source/toolchain, following the
[Rust PGO workflow](https://doc.rust-lang.org/rustc/profile-guided-optimization.html).
`build_pgo.py --target ... --threads ...` supports native CI target selection;
`aarch64-apple-darwin` was exercised in the first pass. The script reserves
additional LLVM value-profile counters because the pool's indirect calls filled
the default reserve. It instruments weft through `cargo rustc`; dependencies
are not separately instrumented. Benchmark the actual CI artifact.

Frinkiac's legacy MP4 endpoint returned HTTP 410, so the fetcher uses the render
API. After an initial HTTP 429, render requests were spaced at least 20 seconds
apart, with one-minute retry backoff. No additional throttle responses occurred.
A stalled download completed on retry. The continued pass made no further
Frinkiac requests.

Final measurements are in `bench/out/arm64/round2/verified-*.json`. The main
comparison is `verified-final.json`; `verified-threads.json` and
`verified-memory.json` record the additional checks. The final executables are
`final-source` and `final-pgo` in that directory. Earlier `final.json`,
`optimized` and `pgo` there describe the rejected fixed-shift prototype.
First-pass artifacts remain in `bench/out/arm64/`. This is a measured hill
climb, without a claim that all ARM64 performance headroom has been exhausted.
