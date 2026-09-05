# Cascade Lake performance hill climb — 2026-09-05

Baseline: `13b87d753eb453df358b41e19249bdb1ab72d8d6`, including the previous hill climb below.
Host: Linux x86-64 KVM, 40 exposed CPUs, Intel Xeon Gold 6248 model (Cascade Lake), AVX-512F/BW/DQ/VL and VNNI. Rust 1.98.1 / LLVM 22.1.8. Both binaries use the repository release profile and `x86-64-v3` target; the new kernels select AVX-512F at runtime.

Flags: `--lossy 30 --dither auto --hold 12 --fps 24 --threads 40 --stats`.
Fourteen fixed, predecoded RGBA clips: ten animation clips, two live-action clips, and two synthetic clips. Each binary gets one warm-up per clip, then **15 runs per clip**, in deterministically shuffled order. The table reports median complete process wall time, including startup, reading cached input, encoding, writing the GIF, and process teardown. Decode time is excluded. Inputs and outputs live in the ignored `gif_bench/` directory. These results describe this host and corpus.

Every measured GIF has exactly the same SHA-256 as baseline. Palette colors, dither decisions, lossy decisions, transparency, frame timing, and file sizes are unchanged on the corpus.

| Clip | Size | Baseline ms | Optimized ms | Throughput gain |
|---|---:|---:|---:|---:|
| old | 480×360 | 149.3 | 144.0 | 3.6% |
| cel | 480×360 | 145.2 | 142.5 | 1.9% |
| modern | 480×270 | 123.1 | 112.5 | 9.4% |
| modern2 | 480×270 | 112.8 | 106.4 | 6.1% |
| grain | 480×360 | 131.4 | 124.9 | 5.1% |
| wide | 720×404 | 168.6 | 161.4 | 4.4% |
| caption | 480×270 | 99.7 | 95.8 | 4.0% |
| old2 | 480×360 | 152.8 | 145.4 | 5.1% |
| short | 480×360 | 83.3 | 77.5 | 7.5% |
| long | 480×270 | 125.1 | 121.3 | 3.1% |
| gradient | 480×270 | 80.8 | 81.8 | -1.3% |
| motion | 480×270 | 81.0 | 75.7 | 7.0% |
| live1 | 480×270 | 110.4 | 101.4 | 8.9% |
| live2 | 480×270 | 128.5 | 123.3 | 4.2% |

Geometric mean throughput: **1.049×** (4.9% faster; 4.7% less wall time). Thirteen clips improved; the low-color synthetic gradient was 1.3% slower. Exploratory seven-run comparisons of the retained palette approach measured about 4–6% overall improvement.

## Cumulative gain from the branch's original hill-climb baseline

A separate three-way comparison on this same Cascade Lake host measures the full branch history: original baseline `caa6927b28457187fa81a30cc40067fd757825f6`, prior optimization commit `13b87d753eb453df358b41e19249bdb1ab72d8d6`, and the current build including the AVX-512 palette changes. All three use Rust 1.98.1, the same release profile and `x86-64-v3` target, the same 14 RGBA inputs, and the same production flags at 40 workers. Each binary gets one warm-up per clip and 15 timed runs in shuffled order (630 timed observations).

| Clip | Original ms | Prior work ms | Current ms | Cumulative throughput gain |
|---|---:|---:|---:|---:|
| old | 141.1 | 139.3 | 135.1 | 4.4% |
| cel | 138.0 | 139.8 | 134.4 | 2.7% |
| modern | 115.9 | 117.0 | 107.4 | 7.9% |
| modern2 | 109.0 | 108.9 | 101.7 | 7.1% |
| grain | 125.4 | 126.3 | 122.8 | 2.1% |
| wide | 163.3 | 159.0 | 154.0 | 6.0% |
| caption | 101.0 | 100.2 | 92.3 | 9.4% |
| old2 | 151.2 | 147.1 | 139.8 | 8.1% |
| short | 83.2 | 83.6 | 80.5 | 3.4% |
| long | 126.3 | 119.4 | 115.3 | 9.5% |
| gradient | 78.6 | 80.3 | 81.0 | -2.9% |
| motion | 84.1 | 80.8 | 75.0 | 12.1% |
| live1 | 104.5 | 105.0 | 99.5 | 5.0% |
| live2 | 123.5 | 121.4 | 116.4 | 6.0% |

Geometric-mean throughput relative to the original baseline is **1.057×**: **5.7% faster**, or **5.4% less wall time**. Within this three-way run, the prior work contributes 1.009× (0.9%), and the new AVX-512 work adds 1.048× (4.8%). Every GIF is byte-identical across all three versions.

This cumulative result is measured directly on the current CPU; the older host's 14.0% result does not establish the prior work's gain on this host. The original source snapshot and its build directory are also under ignored `gif_bench/`; the active checkout was not changed to build it. Raw results: `gif_bench/results/cumulative.json`.

```bash
RUNS=15 python3 gif_bench/measure.py cumulative original baseline optimized
```

## Retained changes

- Add palette kernels requiring only AVX-512F. `fearless_simd` 0.7.0's AVX-512 backend requires Ice Lake features absent on this Cascade Lake host, so the existing generic kernels selected AVX2 here. The separately dispatched kernels cover squared palette distances, nearest-color selection for Lloyd refinement, and candidate-list filtering.
- Keep Lloyd's nearest distance and palette index in vector registers, avoiding the distance-buffer write and subsequent scalar search. Strict comparisons preserve the first index within each lane; the final masked reduction chooses the lowest palette index across tied lanes.
- Filter 16 candidate distances per comparison, visiting set mask bits in ascending order. Candidate lists retain their order and inclusive threshold.
- Keep short palettes on the existing path (fewer than 64 padded entries for distance/nearest selection, or 64 entries for candidate filtering). This cutoff performed better than using the wider kernels for every palette.

The distance kernels retain the existing multiply/add order without FMA contraction. The previous AVX2/NEON paths remain available, and the global build target remains `x86-64-v3`. No worker-count, buffering, palette-quality, or compression-threshold changes were retained.

Phase timings support the overall gain: averaging per-clip medians, palette + LUT time fell from **24.4 to 20.1 ms** (17.7%); Lloyd refinement fell from **6.47 to 3.63 ms** on the 13 clips that need it (43.8%); nearest-map construction fell from **9.28 to 7.75 ms** (16.5%).

## Resource measurements

Five shuffled runs per binary on four clips, without `--stats`, using `/usr/bin/time`. CPU is aggregate user + system seconds; RSS is process peak MiB.

| Clip | CPU seconds before → after | Peak RSS MiB before → after |
|---|---:|---:|
| old | 2.75 → 2.64 | 96.7 → 98.8 |
| wide | 2.98 → 2.86 | 152.2 → 149.8 |
| live1 | 1.96 → 1.92 | 75.1 → 77.1 |
| long | 2.51 → 2.28 | 106.7 → 106.8 |

The retained implementation adds no frame buffers or worker pools. Peak RSS varies by about ±2.4 MiB in these measurements. Hardware counters were unavailable because this host has `perf_event_paranoid=4`; no cycle-count claim is made.

## Rejected experiments

Tested and removed: a Cascade Lake AVX-512 hold kernel; one- and two-worker hold variants; minimum histogram batches of 4, 8, and 16 frames; early rejection of empty LZW candidate masks; narrowing repeated frame-boundary searches; and parallel summation of the two child median-cut boxes. None added a consistent end-to-end gain. A whole-program `target-cpu=native` build also lost to the targeted palette approach (about 2.4% in their seven-run comparison). Binaries and raw measurements remain in ignored `gif_bench/bin/` and `gif_bench/results/`.

## Validation and reproduction

- `cargo fmt --check`
- `cargo clippy --release --all-targets -- -D warnings`
- `cargo test --release`: **62 unit tests and 11 integration tests pass**.
- New SIMD tests compare distances and nearest indices against the generic backend across padded palette sizes and unaligned output slices; exercise tied nearest colors across lanes; and compare candidate lists at every length from 0 through 255, including exact boundaries, infinities, NaNs, and preexisting output prefixes.
- Production-flag integration coverage now includes 40 workers, alongside 1, 8, and 22.
- **34 additional baseline/optimized comparisons** pass: RGBA across 1, 8, 20, and 40 workers; animation and live-action Y4M across 1, 8, and 40; changing alpha at both sides of the transparency threshold across all five dither modes; and smoothing with hold.
- FFmpeg independently decoded all 14 final optimized GIFs successfully.
- Five-run spot benchmarks on `old`, `long`, and `live1`: **9.1%** geometric-mean throughput gain with one worker (individual gains 6.1–14.9%); **3.7%** with eight workers (2.5–5.1%). Outputs match baseline.

The harness was copied from `~/cghmc/scripts/gif_bench/`. Its Go entry point depends on cghmc packages and `/video`, so local Python drivers run weft directly. The entire copied directory, including drivers, source inputs, GIFs, binaries, manifests, and measurements, is gitignored and uncommitted. No media downloads were needed.

With release binaries saved as `gif_bench/bin/baseline` and `gif_bench/bin/optimized`:

```bash
python3 gif_bench/prepare.py
python3 gif_bench/extend.py
RUNS=15 python3 gif_bench/measure.py final baseline optimized
python3 gif_bench/resources.py
python3 gif_bench/verify.py
THREADS=1 RUNS=5 CLIPS=old,live1,long python3 gif_bench/measure.py threads1 baseline optimized
THREADS=8 RUNS=5 CLIPS=old,live1,long python3 gif_bench/measure.py threads8 baseline optimized
```

The ten animation windows match the named episode/start pairs below (`old`, `cel`, `modern`, `modern2`, `grain`, `wide`, `caption`, `old2`, `short`, `long`), decoded from `/mnt/scratch/frinkiac/videos/` with 24 fps and Lanczos scaling to the listed dimensions. `short` contains 36 frames and `long` 156; the other clips contain 96. This is a newly prepared corpus, not the exact cached inputs from the older host: `caption` uses a local drawtext overlay, `gradient` fixes seed 17, and two locally available live-action clips replace `park` and `town`:

- `live1`: first 4 seconds of `/mnt/scratch/weft-corpus-ww/01_S02E19_8s.y4m`.
- `live2`: first 4 seconds of `/mnt/scratch/weft-corpus-ww/03_S01E21_10s.y4m`.
- `gradient`: `gradients=size=480x270:rate=24:speed=0.2:seed=17:duration=4`.
- `motion`: `testsrc2=size=480x270:rate=24:duration=4`.

Full ffmpeg commands, dimensions, frame counts, source paths, and input SHA-256 hashes are in `gif_bench/data/manifest.json`; the final 420 timed observations are in `gif_bench/results/final.json`.

---

# Production-flag performance hill climb — 2026-09-05

Baseline: `caa6927b28457187fa81a30cc40067fd757825f6` (weft 0.4.2).
Host: Linux x86-64 KVM, 22 exposed CPUs, Xeon E5-2696 v4 model, AVX2. Both binaries use the repository release profile and x86-64-v3 target.

Flags: `--lossy 30 --dither auto --hold 12 --fps 24 --threads 22 --stats`.
Fourteen fixed clips, mostly 4 seconds / 96 frames; `short` is 1.5 seconds / 36 frames and `long` is 6.5 seconds / 156 frames. RGBA frames were decoded once with ffmpeg and reused unchanged. Seven runs per binary per clip, in deterministically shuffled order; table reports median complete process wall time, including startup, reading, encoding, and GIF writing. Decode time is excluded. Input files were cached. These measurements describe this host and corpus, not a universal speedup.

Every GIF in the final comparison has exactly the same SHA-256 as baseline. Thus decoded quality, frame timing, transparency, compression decisions, and file size are unchanged on the corpus. No palette, lossy, dither threshold, or hold threshold was relaxed.

| Clip | Size | Baseline ms | Optimized ms | Throughput gain |
|---|---:|---:|---:|---:|
| old | 480×360 | 189.7 | 166.0 | 14.3% |
| cel | 480×360 | 174.5 | 149.9 | 16.4% |
| modern | 480×270 | 136.8 | 125.8 | 8.8% |
| modern2 | 480×270 | 131.3 | 120.8 | 8.7% |
| grain | 480×360 | 160.4 | 137.2 | 16.9% |
| wide | 720×404 | 210.2 | 174.0 | 20.8% |
| park | 480×270 | 223.9 | 193.4 | 15.8% |
| town | 480×270 | 149.5 | 128.7 | 16.1% |
| caption | 480×270 | 111.3 | 99.6 | 11.8% |
| old2 | 480×360 | 175.3 | 160.3 | 9.4% |
| gradient | 480×270 | 70.1 | 61.9 | 13.4% |
| motion | 480×270 | 91.4 | 73.6 | 24.3% |
| short | 480×360 | 82.1 | 79.3 | 3.6% |
| long | 480×270 | 158.0 | 135.1 | 17.0% |

Geometric mean throughput: **1.140×** (14.0% faster; 12.2% less wall time).

## Retained changes

- Process disjoint pixel ranges of each held frame on a dedicated pool of up to four workers. Frames remain sequential, and chunk boundaries align to the 64-pixel histogram sampling period. Integer histogram reduction preserves the adaptive threshold exactly. Small images and fewer than eight configured workers retain serial hold.
- Skip the second auto-dither pass for opaque rows with no live dither tiles and no requested activity scale. Restore the preceding source row when activity processing resumes.
- Increase the quantize/encode block from four to eight frames per worker. Typical short clips then avoid a small, poorly utilized tail block. The working set remains bounded independently of clip length.

## Resource tradeoff

Five shuffled measurements per binary, without `--stats`, using `perf stat` and `/usr/bin/time`. CPU is aggregate user + system seconds; RSS is process peak MiB. Hardware user-space cycle counts fell by approximately 2–4% on these cases.

| Clip | CPU seconds before → after | Peak RSS MiB before → after |
|---|---:|---:|
| old | 2.27 → 2.24 | 89.9 → 97.8 |
| wide | 2.45 → 2.32 | 138.2 → 153.3 |
| park | 3.31 → 3.20 | 77.2 → 83.6 |
| long | 1.93 → 1.92 | 94.2 → 101.6 |

The larger block and additional hold workers increase peak memory by about 6–15 MiB on these samples. Hold uses at most four additional pool workers, separate from the existing histogram/quantization pool.

## Rejected experiments

Tested and removed: tail-calling exact LZW continuations; independent hold histogram lanes; a specialized AVX2 hold kernel; caching the first LZW trie child; AVX2 palette partition compaction; a four-frame minimum histogram batch; and bulk reuse of nearest-color lookups for flat runs. None produced a consistent additional end-to-end win across the corpus. Experiment binaries and results remain under the ignored benchmark directory.

## Validation and reproduction

- `cargo fmt --check`
- `cargo clippy --release --all-targets -- -D warnings`
- `cargo test --release`: 59 unit tests and 11 integration tests pass.
- Added regression coverage for skipped dither rows followed by live rows, and production-flag output across 1, 8, and 22 workers with changing alpha, hold noise, and non-vector-aligned frame dimensions.
- One-worker spot check (old animation, park, short clip; three runs each): 0.5–1.4% faster and identical GIFs.
- Eight-worker spot check (old animation, modern animation, park, short, long; five runs each): 3.3–8.4% faster and identical GIFs.
- Y4M checks on animation and natural motion: identical baseline/optimized output across 1, 8, and 22 workers.
- FFmpeg independently decoded all 14 optimized GIFs successfully.

The copied `gif_bench/` is gitignored, including source media, generated inputs, GIFs, binaries, and raw measurements. Its original Go harness imports cghmc packages and expects `/video`; local Python drivers exercise weft directly on the same kind of predecoded RGBA input. No downloads were necessary because a varied corpus was already available locally.

```bash
python3 gif_bench/prepare.py
python3 gif_bench/extend.py
python3 gif_bench/lengths.py
RUNS=7 python3 gif_bench/measure.py final baseline optimized
python3 gif_bench/resources.py
python3 gif_bench/verify.py
```

Input provenance (full paths and filters in `gif_bench/data/manifest.json`):

- `old`: `S01E07.mkv`, start 125 s.
- `cel`: `S05E03.mkv`, start 310 s.
- `modern`: `S32E03.mkv`, start 240 s.
- `modern2`: `S37E08.mkv`, start 520 s.
- `grain`: `S06E15.mkv`, start 630 s.
- `wide`: `S27E21.mkv`, start 410 s.
- `park`: `mezz_park_joy_1080p50.mkv`, start 2 s.
- `town`: `mezz_old_town_cross_1080p50.mkv`, start 2 s.
- `caption`: `S36E17.mkv`, start 220 s; caption overlay.
- `old2`: `S02E22.mkv`, start 870 s.
- `gradient`: `gradients=size=480x270:rate=24:speed=0.2:duration=4`, start 0 s.
- `motion`: `testsrc2=size=480x270:rate=24:duration=4`, start 0 s.
- `short`: `S05E18.mkv`, start 530 s.
- `long`: `S33E03.mkv`, start 360 s.
