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
--dither D         sierra2 | fs | bluenoise | bayer | none
                   (default: sierra2)
--loop N           loop count, 0 = forever    (default: 0)
--lossy N          lossy LZW compression, 0-200 (default: 0 = lossless
                   encoding of the quantized frames; ~30 is subtle and
                   much smaller on dithered content)
--no-loop          play once (no NETSCAPE extension)
--threads N        worker threads             (default: all cores)
--stats            print timing breakdown to stderr
```

Input is buffered in memory (a single global palette needs two passes), so
peak memory scales with clip size: y4m frames are kept in their native
planar form (e.g. 1.5 bytes/pixel for 4:2:0) and converted on the fly.

## Benchmarks

640×360 @30fps, 150 frames (5 s) except `big` (1280×720, 300 frames);
4 cores; ffmpeg 6.1.1 running its single-command in-memory two-pass
`split[a][b];[a]palettegen[p];[b][p]paletteuse`. Best of 2 runs. PSNR/SSIM
measured on the decoded GIF against the decoded source at the source frame
rate.

| clip      | encoder | time (s) | peak RSS (MB) | size (KB) | PSNR (dB) | SSIM  |
|-----------|---------|---------:|--------------:|----------:|----------:|------:|
| rgba¹     | ffmpeg  | 1.69     | 202           | 2296      | 40.07     | 0.9847|
| rgba¹     | weft    | **0.40** | 152           | **2294**  | **40.08** | 0.9846|
| testsrc   | ffmpeg  | 1.63     | 200           | 2296      | 40.07     | 0.985 |
| testsrc   | weft    | **0.39** | 69            | **2279**  | 39.71²    | 0.899²|
| big (720p)| ffmpeg  | 10.16    | 1152          | 13751     | 42.16     | 0.993 |
| big (720p)| weft    | **2.40** | 415           | **13738** | 41.25²    | 0.900²|
| mandel    | ffmpeg  | 6.22     | 264           | 16762     | 31.40     | 0.886 |
| mandel    | weft    | **0.97** | 93            | **16576** | 31.14     | 0.875 |
| gradients | ffmpeg  | 1.30     | 190           | 2569      | inf³      | 1.000 |
| gradients | weft    | **0.31** | 79            | **2507**  | 43.8²³    | 0.9997|
| life      | ffmpeg  | 1.17     | 190           | 2559      | 76.02     | 0.9997|
| life      | weft    | **0.30** | 70            | **2481**  | 54.8²     | 0.993 |
| static    | ffmpeg  | 1.00     | 189           | 15        | inf³      | 1.000 |
| static    | weft    | **0.27** | 73            | **11**    | 49.7²³    | 0.976 |

**3.4–6.4× faster, smaller output on every clip, ~⅓ the memory.**

¹ `rgba` is the apples-to-apples row: identical input bytes for both
encoders, no YUV→RGB conversion anywhere. Quality is statistically
identical (Δ0.01 dB / Δ0.00003 SSIM).

² The y4m rows undercount weft's quality: the PSNR/SSIM reference is
decoded with swscale, which *truncates* in YUV→RGB where weft rounds to
nearest (e.g. Y=180 → 1.164×164 = 190.96: correctly 191, swscale says
190). weft's output therefore shows a constant ±1–2 offset against that
reference in flat regions — despite being the *closer* conversion — which
caps measurable PSNR around 44–55 dB and dents SSIM in zero-variance
areas. Measured dither speckle (high-pass error RMS) is identical to
ffmpeg's: 2.69/1.52/1.69 vs 2.65/1.52/1.86 per RGB channel on `testsrc`.

³ Sources with ≤255 distinct colors get a bit-exact (lossless) palette
from both encoders.

Reproduce with:

```sh
bench/gen_inputs.sh   # synthesize test clips (needs ffmpeg)
bench/run.sh          # full comparison table
```

## Dithering modes

`sierra2` (default) is error diffusion, matching ffmpeg's paletteuse: the
best-looking mode. `bluenoise` is a two-candidate ordered dither against a
64×64 void-and-cluster blue-noise mask: for each pixel it finds the
nearest palette color, and only when quantization error exists, lets the
threshold pick between the two palette colors spanning that error — so
exact matches never dither and flat regions stay clean. Because it has no
serial error-diffusion chain it is ~2.5× faster to quantize (30% faster
end-to-end at 720p), perfectly temporally stable, and usually a bit
smaller, at slightly lower visual quality on gradient-heavy content
(testsrc: 39.2 dB / 0.8993 SSIM vs sierra2's 39.7 dB / 0.8994, 2% smaller
file). `bayer` and `none` complete the range. The mask is generated by
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
   batched). Palette statistics cost ~nothing beyond input I/O.
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
   distance sweep runs on fearless_simd's portable f32 lanes with runtime
   dispatch, so the baseline-CPU static binaries still use AVX2/NEON when
   the machine has it (bit-exact vs the scalar path — no FMA contraction).
   SIMD is deliberately scoped to this stage: error diffusion is a serial
   dependency chain per pixel, LZW is a serial hash walk, and palette
   lookups are gather-bound, so none of them vectorize profitably.
4. **Quantize + dither, parallel per frame.** Sierra-2-4A error diffusion
   by default (ffmpeg's paletteuse default), diffusing sRGB error with
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
