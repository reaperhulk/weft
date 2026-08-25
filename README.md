# weft

A fast, parallel GIF encoder. Reads raw video from stdin — `yuv4mpegpipe`
(auto-detected) or raw RGBA — and writes an animated GIF to stdout.

weft targets the quality and size of ffmpeg's `palettegen`/`paletteuse`
pipeline (the standard way to make good GIFs) while being several times
faster by parallelizing every stage across frames.

## Install

Prebuilt Linux binaries (x86_64 and aarch64) are attached to releases and
built by CI for every push, in two flavors:

- `weft-*-linux-musl` — fully static, linked against musl with mimalloc as
  the allocator, so they run on any Linux distro — any glibc version,
  Alpine, scratch containers — with no runtime dependencies, at the same
  speed as a glibc build. Peak memory runs higher than the glibc build's
  (roughly 1.5–2×): mimalloc holds onto freed pages where glibc's malloc
  returns them promptly.
- `weft-*-linux-gnu` — dynamically linked against glibc, built on Ubuntu
  24.04 (glibc 2.39). CI verifies no symbol newer than GLIBC_2.39 is
  referenced, so they run on Ubuntu 24.04+ and any other distro with
  glibc ≥ 2.39 (Debian 13, Fedora 40, RHEL 10, ...).

If unsure, take the musl binary; it runs everywhere the gnu one does.

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
--dither D         bluenoise | sierra2 | bayer | none
                   (default: bluenoise)
--dither-gate N    activity gate for bluenoise, 0-720 (default: 16;
                   0 = off): busy regions get progressively less dither
--loop N           loop count, 0 = forever    (default: 0)
--lossy N          lossy LZW compression, 0-200 (default: 0 = lossless
                   encoding of the quantized frames; ~30 is subtle and
                   much smaller on dithered content)
--hold N           temporal hold, 0-765 (default: 0 = off): a pixel that
                   stays within N (|dR|+|dG|+|dB|) of its running mean and
                   within 1.5N of its held value keeps that value; ~8-12
                   is invisible and much smaller on compressed-video input
--smooth N         spatial grain filter, 0-765 (default: 0 = off): each
                   pixel becomes the mean of its 5x5 neighbours within N;
                   edges are excluded so outlines stay crisp. ~24 removes
                   film grain and codec noise (which also defeats --hold)
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
| rgba¹     | ffmpeg  | 2.25     | 201           | 2296      | 40.07     | 0.9847|
| rgba¹     | weft    | **0.34** | 118           | **2160**  | **40.51** | **0.9855**|
| testsrc   | ffmpeg  | 2.06     | 199           | 2296      | 40.07     | 0.985 |
| testsrc   | weft    | **0.31** | 63            | **2170**  | 40.07²    | 0.900²|
| big (720p)| ffmpeg  | 18.23    | 1150          | 13751     | 42.16     | 0.993 |
| big (720p)| weft    | **2.28** | 417           | **13117** | 41.77²    | 0.900²|
| mandel    | ffmpeg  | 9.51     | 264           | 16762     | 31.40     | 0.886 |
| mandel    | weft    | **0.68** | 93            | **16130** | **32.20** | 0.873 |
| gradients | ffmpeg  | 1.76     | 189           | 5378      | inf³      | 1.000 |
| gradients | weft    | **0.35** | 78            | **4906**  | 46.7²³    | 0.9993|
| life      | ffmpeg  | 1.62     | 189           | 3511      | 74.59     | 0.9997|
| life      | weft    | **0.21** | 63            | **3389**  | 53.4²     | 0.993 |
| static    | ffmpeg  | 1.28     | 189           | 15        | inf³      | 1.000 |
| static    | weft    | **0.17** | 63            | **11**    | 49.7²³    | 0.976 |

**5–14× faster, smaller output on every clip, 2–3× less memory.**

Each encoder runs its default dither: sierra2 error diffusion for
ffmpeg's paletteuse, blue-noise for weft (temporally stable and smaller;
see below).

¹ `rgba` is the apples-to-apples row: identical input bytes for both
encoders, no YUV→RGB conversion anywhere. With the activity-gated
blue-noise default, weft measures *higher* PSNR and SSIM than ffmpeg on
identical input, with a 6% smaller file; with `--dither sierra2` — the
same algorithm ffmpeg uses — weft measures 40.08 dB / 0.9846:
statistically identical.

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
| testsrc   | ffmpeg + gifsicle | 12.9     | 1975      | 39.17     | 0.940 |
| testsrc   | weft --lossy 30   | **0.49** | 1992      | 39.36²    | 0.875²|
| big (720p)| ffmpeg + gifsicle | 114.4    | 11854     | 41.02     | 0.974 |
| big (720p)| weft --lossy 30   | **3.67** | **11602** | 41.13²    | 0.891²|
| mandel    | ffmpeg + gifsicle | 39.5     | 11734     | 30.91     | 0.854 |
| mandel    | weft --lossy 30   | **1.24** | 12710     | 31.33     | 0.821 |
| gradients | ffmpeg + gifsicle | 10.3     | 1334      | 48.74     | 0.989 |
| gradients | weft --lossy 30   | **1.16** | **1287**  | 44.87²    | 0.990 |

**9–32× faster than the two-tool pipeline**, at sizes between 4% smaller
and 8% larger, and quality within the dither-default and
reference-decoder gaps above (the weft rows inherit footnote ²).

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
diffusion, matching ffmpeg's paletteuse; with the activity gate in place
the blue-noise default now measures at or above it (testsrc: 40.1 dB /
0.8996 SSIM vs sierra2's 39.7 dB / 0.8994, 5% smaller file), though
sierra2 can still read slightly smoother on gradient-heavy content — at
the cost of speed and temporal stability. `bayer` is the cheap ordered
alternative — a fixed 8x8 threshold matrix, so it compresses better and
is more temporally stable than blue noise, at the cost of visible
cross-hatch structure and weaker banding suppression — and `none`
completes the range. Both ordered modes leave pixels the palette already
reproduces exactly untouched, so a source with 255 or fewer colors
encodes losslessly whatever the dither setting. The mask is generated by
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
| gradients | 4906 KB  | **1287 KB**| −74%   | −1.8 dB| 1335 KB              |
| mandel    | 16130 KB | **12710 KB**| −21%  | −0.9 dB| 11667 KB             |
| testsrc   | 2170 KB  | **1992 KB**| −8%    | −0.7 dB| 1858 KB              |
| big (720p)| 13117 KB | **11602 KB**| −12%  | −0.6 dB| 11424 KB             |

¹ gifsicle 1.94 `-O3 --lossy=30` applied to weft's lossless output.

Two structural pieces land alongside it (and improve lossless output too):
each delta frame is encoded both transparency-punched and plain-opaque and
the smaller wins — sparse changes favor punching, while smooth animated
gradients compress far better opaque (this alone took the gradients clip
from 2555 KB to 2136 KB lossless) — and the encoder defers LZW dictionary
clears, keeping a full dictionary alive while its average match length
holds up (gifsicle's EWMA heuristic) instead of resetting at 4096 codes.

## Prefilters: `--hold` and `--smooth`

Compressed-video input carries a few LSB of noise on every pixel of every
frame, and on flat content that noise — not the picture — decides which
of two neighbouring palette entries a pixel lands on. Re-rolled each frame
it turns static fills into per-frame index churn that defeats the delta
encoder. Two optional prefilters, both running as pipeline stages between
the reader and the histogram pass, address this at the source:

- `--hold N` keeps a pixel at its held value while the input stays within
  N of a per-pixel running mean (and within 1.5N of the held value, which
  bounds the lag on slow drifts). Static regions become byte-identical
  across frames and drop out of the delta entirely.
- `--smooth N` replaces each pixel with the mean of the 5x5 neighbours
  within N of it, so grain averages out inside fills while nothing across
  an edge is touched. With the grain gone, far fewer pixels escape the
  hold window, and the palette sees clean fills.

On a grainy 480x360 cartoon clip at `--lossy 30`: 9.10 MB baseline,
6.40 MB with `--hold 8`, 5.59 MB with `--smooth 24 --hold 8`, at equal
PSNR against the source; on a corpus of random cartoon clips the pair is
worth 8-18% with PSNR unchanged or up. Both are cheap: the hold runs on
one thread, the filter on a small pool, and the encode is faster overall
because the delta stage has less to do.

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

Two hot paths are vectorized with fearless_simd behind runtime dispatch
(SSE4.2/AVX2/AVX-512/NEON picked per machine, so the baseline-CPU static
binaries lose nothing), both verified byte-identical to the scalar paths:
the YUV→RGBA row conversion (16 px/iteration through widen/zip, feeding
both the histogram and quantize passes) and the nearest-map distance
sweep. 720p end-to-end: 2.48 s → 2.13 s. Bulk histogram run extension
(8-pixel blocks, guarded by one scalar compare so noisy content skips the
overhead) uses branchless u64 word compares rather than slice equality:
slice `==` lowers to libc `memcmp`, which musl implements as a
byte-at-a-time loop — that single call site made the static binary's
histogram pass up to 6× slower. Verified not to help: sierra2 error diffusion (the
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

Dependencies: rayon and fearless_simd; the musl static build adds
mimalloc.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
