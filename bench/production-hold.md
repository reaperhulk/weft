# Temporal-hold histogram dependencies

`--lossy 30 --hold 12 --dither auto` makes temporal hold part of the input
pipeline's critical path. The sampled noise histogram frequently increments the
same bin for neighboring pixels. Those increments form a dependent load/add/store
chain. Accumulate into four independent histograms and combine their integer
counts after the vector loop to shorten that chain.

This is a portable algorithm change inside the existing SIMD kernel. It adds
4 KiB of temporary stack storage per active hold worker, with no new heap
allocation, unsafe code, architecture-specific intrinsic, compiler tuning or PGO.
The sampled pixels, tail handling, initial histogram contents, adaptive threshold,
mean updates and output pixels retain their existing semantics.

## Measurement

M5 Max, 18 cores, macOS 26.6.2, Rust 1.96.0 / LLVM 22.1.2, ordinary release
builds. The branch targets main `d09af78c70a346aa1cb825fed115f5b3259e69b2`,
which includes the merged PR #57. The PR adds only the histogram optimization.
The initial investigation used the preceding main `eb1646e`; those measurements
are distinguished below.

The nine Frinkiac MP4 excerpts were decoded once to RGBA. The five selections
originally obtained from the random endpoint are included; all nine clips were
used during this investigation, so they are not an independent holdout for this
change. Requests, dimensions and MP4/RGBA hashes are in `production-manifest.json`.
All clips require quantization. The confirmed production flags are exactly:

```
--lossy 30 --hold 12 --dither auto
```

Twenty-five shuffled paired runs per clip, one warmup for each binary/clip,
18 workers per encode, complete-process wall and child CPU time, file input/output.
Decoding and output hashing are outside the timer. No builds, downloads or other
benchmarks ran concurrently. The final run compares current main with this branch.
Speedup is baseline median / candidate median; aggregate speedup is the geometric
mean across clips.

| Clip | Current main ms | This branch ms | Speedup |
| --- | ---: | ---: | ---: |
| S01E01, 2 s | 23.093 | 22.960 | 1.006× |
| S08E14, 5.5 s | 56.108 | 54.721 | 1.025× |
| S18E10, 9 s | 96.334 | 94.574 | 1.019× |
| S35E15, 9.8 s | 42.969 | 42.378 | 1.014× |
| S14E15, 6 s | 36.046 | 35.855 | 1.005× |
| S06E08, 6 s | 44.592 | 43.842 | 1.017× |
| S20E10, 6 s | 30.128 | 29.726 | 1.014× |
| S21E09, 6 s | 30.974 | 31.107 | 0.996× |
| S19E13, 6 s | 39.651 | 38.740 | 1.024× |

Aggregate: **1.32% more throughput**, **1.11% less child CPU time**, and **6.63%
less reported hold-stage time**. Eight of nine clip medians improved in wall time;
one was 0.43% slower. This is a modest result, not a uniform per-clip win.

Nine shuffled paired batches of 24 encodes, cycling through the nine clips in
manifest order, at 18 workers per encode:

| Concurrent encodes | Current main ms/encode | This branch ms/encode | Speedup |
| --- | ---: | ---: | ---: |
| 1 | 46.445 | 45.890 | 1.012× |
| 10 | 30.691 | 29.918 | 1.026× |

This oversubscribes the M5 at concurrency 10. Native x86 server throughput,
instruction counts and IPC remain unmeasured. The source is portable, but these
numbers establish only an M5 result.

Before PR #57 merged, a four-binary comparison with 25 pairs per clip measured
1.35% more throughput for the histogram change alone against `eb1646e`, and
0.98% on top of PR #57. Concurrent batches against that earlier main improved
1.5% and 2.1% at concurrency 1 and 10. These historical runs are retained in the
evidence, but the tables above use the updated main baseline. The much larger
quantization improvement from PR #57 is already in that baseline.

## Correctness and rejected experiments

All measured GIF hashes match current main. Earlier standalone and combined
comparisons also matched. YUV420, YUV422 and YUV444 input from a 48-frame real-video excerpt
also matches at 1, 4 and 18 workers with the production flags.

Release tests after rebasing onto current main: 88 unit tests and 13 integration tests pass, with one existing
ignored benchmark. The new scalar-reference test covers repeated frames, noise,
changing alpha, six thresholds, initialized histogram counts, multiple frames,
and lengths around SIMD/sampling boundaries, including empty input and tails.
All-target release clippy, formatting and whitespace checks pass.

Discarded: direct single-child lossy dictionary traversal was approximately
neutral; short batches in the hold pipeline regressed end-to-end performance;
a NEON rewrite improved isolated hold timing but did not consistently improve
on the portable histogram change. Reset shortcuts and special handling for
zero-only histogram blocks did not beat the retained approach consistently.

## Reproduction

Place the manifest beside `<name>.rgba` files matching its hashes, decoded from
the recorded MP4 requests with FFmpeg 7.1 and no resizing or frame-rate conversion.
See `fetch_frinkiac.py` for the render endpoint and request pacing.

```sh
python3 bench/arm64.py --binary main=/path/to/main-weft \
  --binary candidate=/path/to/candidate-weft --threads 18 --runs 25 \
  --args='--lossy 30 --hold 12 --dither auto' \
  --manifest /path/to/data/manifest.json --output bench/out/production.json
python3 bench/concurrency_bench.py --binary main=/path/to/main-weft \
  --binary candidate=/path/to/candidate-weft --threads 18 --runs 9 \
  --jobs 24 --concurrency 1,10 --args='--lossy 30 --hold 12 --dither auto' \
  --manifest /path/to/data/manifest.json --output bench/out/production-concurrency.json
```

The runners retain per-iteration wall/CPU times, GIF hashes, stage stats and binary
hashes. The external evidence bundle includes `final-current-main.json`,
`concurrency-current-main.json`, historical `final.json` and `concurrency.json`,
`yuv-correctness.json`, the runners, manifest and test output. No media or executable
files are included in the repository or evidence bundle.
