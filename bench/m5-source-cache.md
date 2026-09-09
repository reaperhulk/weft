# Shared source-color classification

Quantized video repeatedly asks for the nearest palette color of the same RGB
values, across frames and workers. Classifying those values once avoids hash
collisions and duplicated conversion/search work. A shared 16 MiB byte array
holds exact palette indices, populated before quantization from the histogram.
Folded histograms identify occupied 4×4×4 RGB cells; an eight-query portable SIMD
kernel classifies their integer colors using conservative candidate bounds.
An exact balanced OkLab kd-tree handles uncached colors, including dither queries,
and publishes their results atomically. There are no new ARM-specific intrinsics.

This is a standalone change from main `eb1646eae95facfdd1b487098cbe866e1d46749b`.
It does not include the compact exact-palette grid or const-generic refactor from
withdrawn PR #56. Inputs already fitting the palette retain the original lookup.
Sierra2 also retains that lookup: an earlier shared-cache experiment had mixed
Sierra2 performance, including a per-clip regression.

## M5 Max results

Apple M5 Max, 18 cores, macOS 26.6.2, Rust 1.96.0 / LLVM 22.1.2. Both binaries
used ordinary `cargo build --release`, with no PGO or compiler tuning.
These are measurements on this machine, not predictions for an x86 server.

Nine Frinkiac MP4 excerpts were decoded once to RGBA using FFmpeg 7.1, with no
resizing or frame-rate conversion. Four were development clips; the final five
were fresh selections from Frinkiac's random endpoint, retained without filtering
by performance. The source-classification approach was frozen before selecting
those five. All nine require quantization: 36,343–115,979 histogram entries for
255 opaque palette colors. The manifest records render requests and content hashes.

Default encoding, 18 worker threads, one warmup per binary/clip, 15 shuffled paired
runs per clip. Measurements include complete process startup and file input/output;
MP4 decoding and output hashing are outside timing. No builds or other benchmarks
ran concurrently. Speedup means main median time / candidate median time.

| Clip | Main ms | Candidate ms | Speedup |
| --- | ---: | ---: | ---: |
| S01E01, 2.0 s | 19.82 | 17.33 | 1.144× |
| S08E14, 5.5 s | 50.78 | 40.28 | 1.261× |
| S18E10, 9.0 s | 60.24 | 51.83 | 1.162× |
| S35E15, 9.8 s | 39.23 | 33.55 | 1.169× |
| Random S14E15, 6 s | 34.38 | 29.52 | 1.164× |
| Random S06E08, 6 s | 40.19 | 34.19 | 1.175× |
| Random S20E10, 6 s | 30.37 | 25.46 | 1.193× |
| Random S21E09, 6 s | 31.22 | 26.28 | 1.188× |
| Random S19E13, 6 s | 36.99 | 31.45 | 1.176× |

Geometric mean speedup: **1.181×** across nine clips, **1.179×** on the five
fresh clips. This is 15.3% less wall time, or 18.1% more encodes per unit time.
Geometric mean child CPU-time ratio is 1.224×, equivalent to 18.3% less CPU time.
Instructions and IPC were not measured. Both setup and per-frame quantization
improve; the change does not depend solely on nearest-map construction time.

Five shuffled paired batches of 24 encodes, cycling through the nine clips in
manifest order, with 18 workers per encode:

| Concurrent encodes | Main ms/encode | Candidate ms/encode | Speedup |
| --- | ---: | ---: | ---: |
| 1 | 40.09 | 33.51 | 1.197× |
| 10 | 26.79 | 21.82 | 1.228× |

This oversubscribes this host at concurrency 10. It does not substitute for
measurements on a high-core-count x86 server. Sierra2, using the original lookup,
was approximately neutral: 52.15 → 51.87 ms geometric mean, seven pairs per clip.

## Correctness and cost

Every measured default, Sierra2 and concurrent output matched main byte-for-byte.
Additional screens covering no dither, blue noise, Bayer, lossy encoding, hold and
smoothing also matched. There is no visual change to review in these outputs.

`cargo test --release`: 87 unit tests and 13 integration tests pass; one existing
pool benchmark is ignored. New tests compare cold, warm and coarse-cell queries,
small/full palettes, duplicate ties, the maximum opaque palette index, and racing
cache misses. SIMD conversion is checked bit-for-bit against the existing query
conversion for all 16,777,216 RGB colors on the active M5 backend.
`cargo clippy --release --all-targets -- -D warnings` and formatting checks pass.
Native x86-64 and ARM64 Linux glibc/musl test and build jobs also pass.
A 48-frame real-video excerpt through YUV420, YUV422 and YUV444 input at
18, 64 and 256 requested colors produced identical GIF bytes in all nine cases.

The lookup cache occupies 16 MiB per encode. It replaces the grid and per-worker
memo allocations for this strategy, but process peak memory still increased on
most clips. Three measurements per clip, taking each binary's median child peak
RSS, showed changes from −1.8 to +17.2 MiB. This needs particular attention under
server concurrency. Native x86 performance, including instruction counts, CPU
time and concurrent throughput, remains unmeasured; keep the PR in draft
until those are checked. Portable SIMD availability alone is not evidence of an
x86 speedup, and the contributions of SIMD versus caching have not been isolated
in a final-binary ablation.

## Reproduction

`source-cache-manifest.json` contains the exact render requests and MP4/RGBA
SHA-256 hashes. Render each request using the same endpoint and pacing as
`fetch_frinkiac.py`, then decode with:

```sh
ffmpeg -v error -i clip.mp4 -map 0:v:0 -fps_mode passthrough \
  -f rawvideo -pix_fmt rgba clip.rgba
```

Name each RGBA file `<name>.rgba` using the manifest name, and place the manifest
beside those files. Different FFmpeg builds may produce different decoded bytes;
verify the recorded hashes before comparing with these timings.

```sh
python3 bench/arm64.py --binary main=/path/to/main-weft \
  --binary candidate=/path/to/candidate-weft --threads 18 --runs 15 \
  --manifest /path/to/data/manifest.json --output bench/out/source-cache.json
python3 bench/concurrency_bench.py --binary main=/path/to/main-weft \
  --binary candidate=/path/to/candidate-weft --threads 18 --runs 5 \
  --jobs 24 --concurrency 1,10 --manifest /path/to/data/manifest.json \
  --output bench/out/source-cache-concurrent.json
```

Despite its historical name, `arm64.py` uses no architecture-specific measurement
API and can run on Linux. Both runners retain per-iteration timings, output hashes,
stats, input metadata and binary hashes. The external evidence bundle contains
`standalone-final.json`, `concurrent-final.json`, `sierra-final.json` and
`memory-final.json`; media and executable files are excluded.
