# weft

A fast, parallel GIF encoder. Reads raw video from stdin — `yuv4mpegpipe`
(auto-detected) or raw RGBA — and writes an animated GIF to stdout.

weft targets the quality and size of ffmpeg's `palettegen`/`paletteuse`
pipeline (the standard way to make good GIFs) while being several times
faster by parallelizing every stage across frames.

## Install

Prebuilt fully static Linux binaries (x86_64 and aarch64) are attached to
releases and built by CI for every push. They are linked against musl with
mimalloc as the allocator, so they run on any Linux distro — any glibc
version, Alpine, scratch containers — with no runtime dependencies, at the
same speed as a glibc build.

From source: `cargo build --release` (or
`cargo build --release --target x86_64-unknown-linux-musl` for the static
build; needs `musl-tools`).

## Usage

```sh
# from any video, via ffmpeg's decoder
ffmpeg -i input.mp4 -f yuv4mpegpipe - | weft > out.gif

# raw RGBA frames (e.g. from a renderer)
my-renderer | weft --size 640x360 --fps 30 > out.gif

# fewer colors, no dithering, play once
weft --colors 64 --dither none --no-loop < input.y4m > out.gif
```

```
--size WxH         frame size (required for raw RGBA input)
--fps N[/D]        frame rate (raw RGBA default: 30; overrides y4m header)
--format F         auto | rgba | y4m          (default: auto)
--colors N         max palette colors, 2-256  (default: 256; one slot is
                   reserved for transparency, so 256 means 255 colors)
--dither D         bluenoise | sierra2 | fs | bayer | none
                   (default: bluenoise)
--dither-gate N    activity gate for bluenoise, 0-720 (default: 16;
                   0 = off): busy regions get progressively less dither
--loop N           loop count, 0 = forever    (default: 0)
--lossy N          lossy LZW compression, 0-200 (default: 0 = lossless
                   encoding of the quantized frames; ~30 is subtle and
                   much smaller on dithered content)
--no-loop          play once (no NETSCAPE extension)
--threads N        worker threads             (default: all cores)
--stats            print timing breakdown to stderr
```

Input is buffered in memory (a single global palette needs two passes), so
peak memory scales with clip size. Frames are kept in the smallest form
that reproduces their pixels exactly and converted to RGBA rows on the fly:
y4m frames stay in their native planar form (e.g. 1.5 bytes/pixel for
4:2:0), and raw RGBA frames whose pixels are all opaque — the overwhelmingly
common case — drop their constant alpha byte and are stored as 3 bytes/pixel.
Frames that do use transparency keep their alpha; the choice is per frame,
so a clip that mixes them pays only for the frames that need it.

## Benchmarks

640×360 @30fps, 150 frames (5 s) except `big` (1280×720, 300 frames);
4 cores; ffmpeg 6.1.1 running its single-command in-memory two-pass
`split[a][b];[a]palettegen[p];[b][p]paletteuse`. Best of 2 runs. PSNR/SSIM
measured on the decoded GIF against the decoded source at the source frame
rate.

| clip      | encoder | time (s) | peak RSS (MB) | size (KB) | PSNR (dB) | SSIM  |
|-----------|---------|---------:|--------------:|----------:|----------:|------:|
| rgba¹     | ffmpeg  | 1.63     | 202           | 2296      | 40.07     | 0.9847|
| rgba¹     | weft    | **0.24** | 37            | **2226**  | 39.60¹    | **0.9853**|
| testsrc   | ffmpeg  | 1.70     | 199           | 2296      | 40.07     | 0.985 |
| testsrc   | weft    | **0.22** | 28            | **2242**  | 39.24²    | 0.899²|
| big (720p)| ffmpeg  | 10.84    | 1152          | 13751     | 42.16     | 0.993 |
| big (720p)| weft    | **1.32** | 75            | **13596** | 41.09²    | 0.900²|
| mandel    | ffmpeg  | 7.12     | 265           | 16762     | 31.40     | 0.886 |
| mandel    | weft    | **0.85** | 108           | **16399** | 30.54     | 0.864 |
| gradients | ffmpeg  | 1.29     | 190           | 2569      | inf³      | 1.000 |
| gradients | weft    | **0.24** | 36            | **2086**  | 43.8²³    | 0.9997|
| life      | ffmpeg  | 1.04     | 190           | 2559      | 76.02     | 0.9997|
| life      | weft    | **0.25** | 31            | **2467**  | 54.8²     | 0.993 |
| static    | ffmpeg  | 1.05     | 190           | 15        | inf³      | 1.000 |
| static    | weft    | **0.12** | 29            | **11**    | 49.7²³    | 0.976 |

**4–9× faster, smaller output on every clip, 5–15× less memory.**

Each encoder runs its default dither: sierra2 error diffusion for
ffmpeg's paletteuse, blue-noise for weft (temporally stable and smaller;
see below).

¹ `rgba` is the apples-to-apples row: identical input bytes for both
encoders, no YUV→RGB conversion anywhere. The 0.5 dB PSNR gap is the
blue-noise default's deliberate trade (SSIM is slightly *higher*, and the
file 3% smaller); with `--dither sierra2` — the same algorithm ffmpeg
uses — weft measures 40.08 dB / 0.9846: statistically identical.

² The y4m rows undercount weft's quality: the PSNR/SSIM reference is
decoded with swscale, which *truncates* in YUV→RGB where weft rounds to
nearest (e.g. Y=180 → 1.164×164 = 190.96: correctly 191, swscale says
190). weft's output therefore shows a constant ±1–2 offset against that
reference in flat regions — despite being the *closer* conversion — which
caps measurable PSNR around 44–55 dB and dents SSIM in zero-variance
areas. With `--dither sierra2` (ffmpeg's algorithm), measured dither
speckle (high-pass error RMS) is identical to ffmpeg's: 2.69/1.52/1.69
vs 2.65/1.52/1.86 per RGB channel on `testsrc`.

³ Sources with ≤255 distinct colors get a bit-exact (lossless) palette
from both encoders.

### Lossy mode

`weft --lossy 30` against the standard two-tool pipeline — the ffmpeg
encode above piped through `gifsicle -O3 --lossy=30` (gifsicle 1.94),
timed as the sum of both stages:

| clip      | pipeline          | time (s) | size (KB) | PSNR (dB) | SSIM  |
|-----------|-------------------|---------:|----------:|----------:|------:|
| testsrc   | ffmpeg + gifsicle | 7.9      | 1975      | 39.17     | 0.940 |
| testsrc   | weft --lossy 30   | **0.29** | 2047      | 38.61²    | 0.874²|
| big (720p)| ffmpeg + gifsicle | 63.5     | 11854     | 41.02     | 0.974 |
| big (720p)| weft --lossy 30   | **1.81** | 11932     | 40.55²    | 0.891²|
| mandel    | ffmpeg + gifsicle | 28.3     | 11734     | 30.91     | 0.854 |
| mandel    | weft --lossy 30   | **1.42** | 12064     | 30.13     | 0.833 |
| gradients | ffmpeg + gifsicle | 4.1      | 663       | 47.63     | 0.989 |
| gradients | weft --lossy 30   | **0.57** | 663       | 42.63²    | 0.990 |

**7–35× faster than the two-tool pipeline**, at sizes within 3% and
quality within the dither-default and reference-decoder gaps above (the
weft rows inherit footnote ²).

Reproduce with:

```sh
bench/gen_inputs.sh   # synthesize test clips (needs ffmpeg)
bench/run.sh          # full comparison table
```

## Dithering modes

`bluenoise` (default) is a two-candidate ordered dither against a 64×64
void-and-cluster blue-noise mask: for each pixel it finds the nearest
palette color, and only when quantization error exists, lets the
threshold pick between the two palette colors spanning that error — so
exact matches never dither and flat regions stay clean. Because it has no
serial error-diffusion chain it is much faster to quantize, perfectly
temporally stable, and usually a bit smaller.

By default blue-noise is also *activity-gated*: per pixel, a cheap local
activity measure (summed channel differences against the left and upper
neighbors) attenuates the dither, at full strength up to `--dither-gate`
and ramping to none over the next 64 units. Smooth gradients — the only
place ordered dither is needed to hide banding — sit far below the gate
and keep full dither, while texture and edges, where palette error is
visually masked and dither reads as churning speckle, degrade to plain
nearest-color. On real-world content this measures better on every axis:
on random frinkiac cartoon clips the default gate is worth +1.2–1.9 dB
PSNR, higher SSIM, ~5–10% smaller files, and ~6–38% faster quantization
(gated pixels skip the far-candidate work; fully gated tiles skip it
wholesale). On pure-gradient content the gate never engages and output is
bit-identical to `--dither-gate 0`. The gate reads only the source frame,
so it is exactly as temporally stable as the ungated dither. `sierra2` is error
diffusion, matching ffmpeg's paletteuse: slightly higher visual quality
on gradient-heavy content (testsrc: 39.7 dB / 0.8994 SSIM vs bluenoise's
39.2 dB / 0.8993, 2% larger file) at the cost of speed and temporal
stability. `bayer` and `none` complete the range. The mask is generated by
`bench/gen_bluenoise.py` (Ulichney void-and-cluster) and checked in as
`src/bluenoise.rs`.

## Lossy compression

`--lossy N` ports gifsicle's `--lossy` algorithm: a DFS over the LZW
dictionary trie finds the longest match whose per-pixel color error stays
under N*10 (squared RGB, gifsicle's scale), with each substitution's
signed error fed forward into the next pixel's comparison at 3/4 decay so
errors cancel instead of accumulating. Dithered content — full of visually
interchangeable palette indices that break long runs — compresses
dramatically better. Measured at `--lossy 30` (same clips as above):

| clip      | lossless | --lossy 30 | Δ size | Δ PSNR | gifsicle --lossy=30¹ |
|-----------|---------:|-----------:|-------:|-------:|---------------------:|
| gradients | 2086 KB  | **664 KB** | −68%   | −1.3 dB| 665 KB               |
| mandel    | 16358 KB | **11656 KB**| −29%  | −0.4 dB| 11614 KB             |
| testsrc   | 2280 KB  | **2084 KB**| −9%    | −0.8 dB| 1944 KB              |
| big (720p)| 13738 KB | **12096 KB**| −12%  |        |                      |

¹ gifsicle 1.94 `-O3 --lossy=30` applied to weft's lossless output.

Two structural pieces land alongside it (and improve lossless output too):
each delta frame is encoded both transparency-punched and plain-opaque and
the smaller wins — sparse changes favor punching, while smooth animated
gradients compress far better opaque (this alone took the gradients clip
from 2555 KB to 2136 KB lossless) — and the encoder defers LZW dictionary
clears, keeping a full dictionary alive while its average match length
holds up (gifsicle's EWMA heuristic) instead of resetting at 4096 codes.

## Design

Every heavy stage is embarrassingly parallel across frames (rayon):

1. **Read + histogram, overlapped.** A reader thread streams frames into a
   bounded channel while workers accumulate per-thread exact-color
   histograms (open-addressed hash keyed by 24-bit color, run-length
   batched). Palette statistics cost ~nothing beyond input I/O. RGBA rows
   are hashed in place, straight out of the frame; only y4m pays a
   conversion into a scratch row. This pass also records, per frame,
   whether any pixel is transparent — the pipeline needs that before
   quantization to pick a disposal mode, and it doubles as the test for
   packing an all-opaque frame down to 3 bytes/pixel for the rest of the
   run.
2. **Palette: variance median cut in OkLab**, mirroring ffmpeg ≥5.x
   palettegen: the box with the largest single-channel squared error (in
   Lab) splits at its count-weighted median along that channel; each box
   emits the Lab average of its colors. Boxes that collapse to (or are
   ≥99% dominated by) one exact color emit that color byte-exactly.
   Sources with ≤255 distinct colors skip straight to a lossless palette.
3. **Exact nearest-color lookup** via per-cell candidate lists over a
   6-bit/channel RGB grid (locally sorted search): a triangle-inequality
   bound makes the argmin over the cell's candidates the true OkLab
   nearest for every color in the cell. Most cells hold one candidate, so
   the hot path is a single table load; multi-candidate lookups memoize in
   a per-thread direct-mapped cache. The build's 262k-cell × 256-color
   distance sweep runs on fearless_simd's portable f32 lanes.

Three hot paths are vectorized with fearless_simd behind runtime dispatch
(SSE4.2/AVX2/AVX-512/NEON picked per machine, so the baseline-CPU static
binaries lose nothing), all verified byte-identical to the scalar paths:
the YUV→RGBA row conversion (16 px/iteration through widen/zip, feeding
both the histogram and quantize passes), the nearest-map distance sweep,
and bulk histogram run extension (8-pixel block compares, guarded by one
scalar compare so noisy content skips the overhead). 720p end-to-end:
2.48 s → 2.13 s. Verified not to help: sierra2 error diffusion (the
carry→lookup→error chain is latency-bound, not throughput-bound), LZW (a
serial hash walk), and palette lookups (gather-bound). Known follow-up: a
radix sort on the cut axis inside median cut (mandel's remaining
hotspot).
4. **Quantize + dither, parallel per frame.** Blue-noise ordered dither
   by default; `--dither sierra2` selects Sierra-2-4A error diffusion
   (ffmpeg's paletteuse default), which diffuses sRGB error with
   truncating division (an arithmetic shift would diffuse >100% of
   negative error and explode into noise). YUV→RGB conversion is fused
   row-by-row into this pass — full RGBA frames never materialize.
5. **Delta + LZW, parallel per frame.** Frames crop to the changed
   bounding box; the rect is encoded both transparency-punched and plain
   opaque, keeping whichever is smaller; identical frames fold their delay
   into the predecessor. With disposal "none" the decoded canvas after
   frame *i−1* equals indexed frame *i−1*, so frame *i*'s delta needs only
   its neighbor — no serial canvas walk. LZW uses a generation-stamped
   open-addressed table (no per-clear reset), a 64-bit bit accumulator,
   and gifsicle-style deferred dictionary clears; `--lossy` adds the
   error-bounded match search described above.
6. **Mux.** Single sequential write of pre-encoded chunks.

Correctness is pinned by unit tests (LZW roundtrip incl. forced clears,
palette exactness, nearest-map-vs-brute-force in OkLab, y4m parsing, delay
accumulation) and an end-to-end test that decodes weft's output with an
independent minimal GIF decoder and compares canvases byte-for-byte.

Only dependency: rayon.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
