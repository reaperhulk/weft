# weft

A fast, parallel GIF encoder. Reads raw video from stdin — `yuv4mpegpipe`
(auto-detected) or raw RGBA — and writes an animated GIF to stdout.

weft targets the quality and size of ffmpeg's `palettegen`/`paletteuse`
pipeline (the standard way to make good GIFs) while being several times
faster by parallelizing every stage across frames.

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
--dither D         sierra2 | fs | bayer | none (default: sierra2)
--loop N           loop count, 0 = forever    (default: 0)
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
   a per-thread direct-mapped cache.
4. **Quantize + dither, parallel per frame.** Sierra-2-4A error diffusion
   by default (ffmpeg's paletteuse default), diffusing sRGB error with
   truncating division (an arithmetic shift would diffuse >100% of
   negative error and explode into noise). YUV→RGB conversion is fused
   row-by-row into this pass — full RGBA frames never materialize.
5. **Delta + LZW, parallel per frame.** Unchanged pixels become
   transparent, frames crop to the changed bounding box, identical frames
   fold their delay into the predecessor. With disposal "none" the decoded
   canvas after frame *i−1* equals indexed frame *i−1*, so frame *i*'s
   delta needs only its neighbor — no serial canvas walk. LZW uses a
   generation-stamped open-addressed table (no per-clear reset) and a
   64-bit bit accumulator.
6. **Mux.** Single sequential write of pre-encoded chunks.

Correctness is pinned by unit tests (LZW roundtrip incl. forced clears,
palette exactness, nearest-map-vs-brute-force in OkLab, y4m parsing, delay
accumulation) and an end-to-end test that decodes weft's output with an
independent minimal GIF decoder and compares canvases byte-for-byte.

Only dependency: rayon.
