# Selective auto dithering on mixed frames

Baseline: Weft `562965dc7a17cf9aee93b2991638563c8ba508dc`.
Harness: cghmc `570dbc121ba2e427f5e77c8c6f18d3931fa77a18`.

Auto dithering detects banding per 32x32 tile, but the previous fallback
promoted the entire frame to dithering once more than 25% of tiles were
flagged. Raising that threshold to 75% retains selective dithering on
mixed frames. The band detector, palette, temporal hold, and lossy error
cap are unchanged. This is an intentional image-output change, not a
byte-identical throughput optimization.

## Inputs and machine

Nine Frinkiac MP4 previews, no overlays, decoded to 480x360 RGBA:
48, 96, or 156 frames per clip. Source windows are the episode/start pairs
below; durations are respectively 4, 6.5, 2, 4, 6.5, 4, 2, 4, 6.5 seconds.
The previews were requested from `/api/render/mp4/stream` with an array
containing `{episode, start, end, overlays: [], preview: true}`, then
retrieved from the returned URL. These previews have an extra video
compression pass relative to production's original video files.

AMD EPYC 9V74 VM, five visible vCPUs, **four-core cgroup quota**.
Rust 1.99.1, `cargo build --release --locked`, repository x86-64-v3 flags,
glibc. This is not cghmc's 40-core Cascade Lake production host.
All timing arms ran sequentially with randomized ordering; no compilation
or quality evaluation ran alongside timed benchmarks.

## Production-setting concurrency results

Unmodified cghmc `scripts/gif_bench/thread_bench.py` with:

```sh
python scripts/gif_bench/thread_bench.py conc RAW_DIR \
  --bins baseline=/path/to/baseline,full75=/path/to/candidate \
  --threads 20 --conc 1,10 --batch 18 --reps 5 --seed 41971
```

Its flags are `--fps 25 --lossy 30 --dither auto --hold 12`; threads are
explicit because the harness's default is not cghmc's current explicit
20-worker production setting. Input is replayed from regular files, so
these measure encoding, not production decode/pipe latency.

Medians of five batches, eighteen encodes per batch:

| Concurrent encodes | Baseline ms/GIF | Candidate ms/GIF | Elapsed change | CPU change |
|---|---:|---:|---:|---:|
| 1 | 344.7 | 332.6 | -3.5% | -4.4% |
| 10 | 300.3 | 291.9 | -2.8% | -2.3% |

At concurrency ten, ms/GIF is batch-drain throughput, not individual
request latency. CPU values are child user+system time, not perf counters.

A separate five-repetition per-clip screen at four workers measured a
median paired quantization-time ratio of 0.802, whole-process CPU ratio
0.929, and wall ratio 0.946. Phase timings came from `--stats`; the
production-setting table above is the primary end-to-end result.

## Decoded image quality and size

FFmpeg decoded each GIF at 25 fps and compared RGB against the identical
raw RGBA input using PSNR and SSIM. Both filter inputs use `setpts=N`.
Values below are candidate minus baseline, except size is percentage change.

| Clip / start ms | PSNR dB | SSIM | GIF bytes |
|---|---:|---:|---:|
| S06E21@600000 | +0.281 | +0.01253 | -5.62% |
| S03E05@300000 | +0.691 | +0.02309 | -5.81% |
| S04E12@900000 | +0.000 | +0.00000 | +0.00% |
| S05E02@450000 | +0.275 | +0.01019 | -6.56% |
| S07E08@750000 | +0.555 | +0.02094 | -5.98% |
| S08E01@180000 | +0.455 | +0.01126 | -10.29% |
| S02E11@1050000 | +0.000 | +0.00000 | +0.00% |
| S10E02@750000 | +0.189 | +0.00835 | -17.09% |
| S06E21@1100000 | +0.436 | +0.01684 | -15.49% |

Median: +0.281 dB PSNR, +0.011264 SSIM, -5.98% bytes. No clip regressed
on these averaged image metrics. Seven additional inputs (four Big Buck
Bunny windows, testsrc2, a gradient, and static bars) also improved or
preserved PSNR/SSIM and did not grow. The gradient and static outputs were
unchanged in size and metrics. These objective averages do not prove the
absence of local or temporal banding on every source. Still-frame spot
checks on two Frinkiac scenes showed less speckling in flat fills without
an obvious new artifact. Validate on the original production corpus and
host before extrapolating throughput or changing a release default.

## Other experiments

Rejected as unproven: overlapping SIMD tails in the band detector,
fusing its passes, packing the nearest-color kd-tree, fast cube roots
for palette construction, and reducing the lossy LZW search budget to
512. Some isolated timings improved, but repeatable end-to-end wins did
not emerge.

A five-bit seed histogram reduced median-cut time substantially and
screened about 8% faster overall, but lost up to 0.01393 SSIM on a Frinkiac
clip. It is not included in this patch.

## Validation

- `cargo test --release --locked`: 108 passed, one ignored benchmark.
- `cargo fmt --check` and release all-target Clippy with `-D warnings`.
- New mixed-frame regression: a genuinely ditherable flat fill stays at
  its nearest colour while a separate banding region is still dithered.
- Existing tests cover thread invariance, transparency, hold, and GIF
  round trips.
