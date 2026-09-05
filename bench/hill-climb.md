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
