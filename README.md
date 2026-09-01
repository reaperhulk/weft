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

The x86_64 binaries require an x86-64-v3 CPU or better (AVX2, FMA, BMI2:
Intel Haswell from 2013, AMD Zen from 2017, or later). On an older
machine they die with an illegal-instruction error at startup; build from
source with `RUSTFLAGS="-C target-cpu=x86-64"` instead. The aarch64
binaries run on any 64-bit ARM.

From source: `cargo build --release` (or
`cargo build --release --target x86_64-unknown-linux-musl` for the static
build; needs `musl-tools`). `.cargo/config.toml` sets `target-cpu` to
x86-64-v3 for x86_64 targets; a `RUSTFLAGS` in the environment overrides
it.

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
--dither D         auto | bluenoise | sierra2 | bayer | none
                   (default: auto — blue noise only in 32x32 tiles whose
                   nearest-colour map shows banding contours, plain
                   nearest colour elsewhere)
--dither-gate N    activity gate for bluenoise, 0-720 (default: 16;
                   0 = off): busy regions get progressively less dither
--loop N           loop count, 0 = forever    (default: 0)
--lossy N          lossy LZW compression, 0-200 (default: 0 = lossless
                   encoding of the quantized frames; ~30 is subtle and
                   much smaller on dithered content)
--hold N           temporal hold, 0-765 (default: 0 = off): a pixel that
                   stays within the hold window of its running mean (and
                   1.5x it of its held value) keeps that value. The window
                   adapts per frame to the measured frame-to-frame noise
                   and N caps it; ~12 is a safe cap, and much smaller on
                   compressed-video input
--smooth N         spatial grain filter, 0-765 (default: 0 = off): each
                   pixel becomes the mean of
                   its 5x5 neighbours within N; edges are excluded so
                   outlines stay crisp. ~16-24 removes film grain and
                   codec noise (which also defeats --hold)
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
40-vCPU Xeon 6248 (KVM guest); ffmpeg 8.0.1-3ubuntu2 running its
single-command in-memory two-pass
`split[a][b];[a]palettegen[p];[b][p]paletteuse`. Best of 3 runs
(`RUNS=3 bench/run.sh`). PSNR/SSIM measured on the decoded GIF against the
decoded source at the source frame rate.

| clip      | encoder | time (s) | peak RSS (MB) | size (KB) | PSNR (dB) | SSIM  |
|-----------|---------|---------:|--------------:|----------:|----------:|------:|
| rgba¹     | ffmpeg  | 1.55     | 214           | 2296      | 40.07     | 0.9847|
| rgba¹     | weft    | **0.12** | 155           | **2024**  | **42.27** | **0.9880**|
| testsrc   | ffmpeg  | 1.54     | 207           | 2296      | 40.07     | 0.9847|
| testsrc   | weft    | **0.11** | 83            | **2047**  | **41.64**²| 0.902²|
| big (720p)| ffmpeg  | 10.02    | 1166          | 13751     | 42.16     | 0.9925|
| big (720p)| weft    | **0.56** | 491           | **12588** | **43.31**²| 0.902²|
| mandel    | ffmpeg  | 6.48     | 273           | 16762     | 31.40     | 0.8858|
| mandel    | weft    | **0.19** | 113           | **16055** | **32.99** | 0.8843|
| gradients | ffmpeg  | 1.13     | 198           | 2340      | 67.13³    | 0.9998|
| gradients | weft    | **0.11** | 91            | **2286**  | 49.17²³   | 0.9993|
| life      | ffmpeg  | 1.08     | 198           | 2947      | 75.41³    | 0.9997|
| life      | weft    | **0.08** | 73            | **2844**  | 54.21²³   | 0.9927|
| static    | ffmpeg  | 1.11     | 198           | 15        | inf³      | 1.000 |
| static    | weft    | **0.08** | 72            | **11**    | 49.74²³   | 0.9762|

**10–34× faster, smaller output on every clip, 1.4–2.8× less memory.**

Each encoder runs its default dither: sierra2 error diffusion for
ffmpeg's paletteuse, `auto` for weft (blue noise where the picture bands,
plain nearest colour elsewhere; see below).

¹ `rgba` is the apples-to-apples row: identical input bytes for both
encoders, no YUV→RGB conversion anywhere. weft measures *higher* PSNR
and SSIM than ffmpeg on identical input, with a 12% smaller file.

² The y4m rows undercount weft's quality: the PSNR/SSIM reference is
decoded with swscale, which *truncates* in YUV→RGB where weft rounds to
nearest (e.g. Y=180 → 1.164×164 = 190.96: correctly 191, swscale says
190). weft's output therefore shows a constant ±1–2 offset against that
reference in flat regions — despite being the *closer* conversion — which
caps measurable PSNR around 44–55 dB and dents SSIM in zero-variance
areas.

³ Sources with ≤255 distinct colors get a bit-exact (lossless) palette
from both encoders; on the y4m rows only the conversion offset of ²
separates them.

### Lossy mode

`weft --lossy 30` against the standard two-tool pipeline — the ffmpeg
encode above piped through `gifsicle -O3 --lossy=30` (gifsicle 1.96),
timed as the sum of both stages:

| clip      | pipeline          | time (s) | size (KB) | PSNR (dB) | SSIM  |
|-----------|-------------------|---------:|----------:|----------:|------:|
| testsrc   | ffmpeg + gifsicle | 7.63     | 2188      | 39.75     | 0.965 |
| testsrc   | weft --lossy 30   | **0.13** | **1869**  | **40.72**²| 0.881²|
| big (720p)| ffmpeg + gifsicle | 60.4     | 13493     | 42.16     | 0.992 |
| big (720p)| weft --lossy 30   | **0.71** | **11009** | **42.42**²| 0.898²|
| mandel    | ffmpeg + gifsicle | 30.9     | 16341     | 31.39     | 0.886 |
| mandel    | weft --lossy 30   | **0.26** | **12546** | **32.01** | 0.834 |
| gradients | ffmpeg + gifsicle | 3.84     | 1594      | 61.64³    | 0.9994|
| gradients | weft --lossy 30   | **0.12** | **770**   | 47.93²³   | 0.9964|

**32–119× faster than the two-tool pipeline, 15–52% smaller on every
clip**, and higher PSNR on three of the four (the fourth is the
few-colour case of footnote ³; the weft rows also inherit footnote ²).

Reproduce with:

```sh
bench/gen_inputs.sh   # synthesize test clips (needs ffmpeg)
bench/run.sh          # full comparison table
```

## Dithering modes

`auto` (default) is `bluenoise` gated per tile by a banding detector —
see below. `bluenoise` is a two-candidate ordered dither against a 64×64
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

`auto`, the default, addresses the case where neither extreme is right: on flat content
(cel animation, UI, text) dither only renders invisible quantization error
as visible speckle and `none` looks better and compresses smaller, while
on gradients (skies, dark scenes, skin) `none` posterizes. `auto` runs the
blue-noise pipeline but decides per 32x32 tile from a banding detector on
the nearest-colour map: a tile is dithered when enough of its pixels sit
on a contour between two long same-colour runs whose colours differ by a
visible-but-small OkLab step — the signature of posterization, which grain
(short runs) and outlines (large steps) fail. A frame that bands in more
than a quarter of its tiles is dithered whole. Tiles that do not band skip
the far-candidate stages entirely, so `auto` costs about the same as
`bluenoise` on gradient content and approaches `none` on flat content.
Measured: cel animation dithers ~10% of tiles, live action 10-35%,
synthetic gradients 100%; size and PSNR land between the two modes.

The detector reads structure off the nearest-colour map, so it needs
that map to reflect the picture rather than the noise: on raw grainy
input a flat fill sitting between two palette entries quantizes into
blobs of both, wide enough with a small enough colour step to pass for
banding — the gate then fires on nearly every tile and `auto` degrades
to plain blue noise (today's behaviour, not worse). On such sources pair
it with `--smooth` (16 is safe on live action, 24 on cel animation);
`--stats` reports the dithered-tile fraction, and a reading near 100% on
content that is not all gradient is the tell. `--smooth` is not implied
by default because it invents colours: a source with fewer colours than
the palette is encoded exactly without it and grows (measured +60% on a
two-colour automaton) with it.

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
| gradients | 2286 KB  | **770 KB** | −66%   | −1.2 dB| 1648 KB              |
| mandel    | 16055 KB | **12546 KB**| −22%  | −1.0 dB| 15743 KB             |
| testsrc   | 2047 KB  | **1869 KB**| −9%    | −0.9 dB| 1931 KB              |
| big (720p)| 12588 KB | **11009 KB**| −13%  | −0.9 dB| 12428 KB             |

weft's own lossy pass now beats post-processing its lossless output with
gifsicle on all four clips, which was not true when this table was first
measured — gifsicle then won on everything but `gradients`.

¹ gifsicle 1.96 `-O3 --lossy=30` applied to weft's lossless output.

With `--dither none`, the error cap is scaled per pixel: full where the
source is busy, falling to ~3% of it in perfectly smooth regions. Lossy
substitution there is what produces the plateaus and hatching that make
undithered gradients look posterized — the encoder's error feedback
toggles between two indices to hold the average, and with no dither
texture to hide in, the toggling shows — while the small residual budget
still allows the invisible swaps between adjacent entries of a smooth
ramp.

`bluenoise` and `auto` keep the flat cap. For `bluenoise` the dither
absorbs the substitutions. For `auto` the reason is less obvious and was
found the hard way: `auto` leaves a region undithered precisely where its
gate found no contour, so scaling the cap down in those regions strips out
the LZW noise that had been masking the contours the gate *missed* — and
it does miss them, at OkLab ΔE below the 0.012 floor it treats as visible.
Keeping the flat cap under `auto` measured both smaller and flatter.

Two structural pieces land alongside it (and improve lossless output too):
each delta frame is encoded both transparency-punched and plain-opaque and
the smaller wins — sparse changes favor punching, while smooth animated
gradients compress far better opaque (this alone took the gradients clip
from 2555 KB to 2136 KB lossless) — and the encoder defers LZW dictionary
clears, keeping a full dictionary alive while its average match length
holds up (gifsicle's EWMA heuristic) instead of resetting at 4096 codes.

## Palette

The global palette is a variance median cut in OkLab over the clip's
colour histogram, followed by three Lloyd (k-means) passes: each
colour moves to the count-weighted mean of the colours that actually map
to it. Median cut places colours at box means, but once the neighbouring
boxes exist a colour's nearest set is not its box; the passes correct
that, and on the measured clips are worth +0.4-0.8 dB mean PSNR (up to
+1.2 dB on the worst frame) at unchanged file size — most visibly in dark
gradients, which median cut alone under-serves. Clusters dominated by a
single colour snap to it exactly, so flat fills and sources with few
colours stay lossless.

The histogram is exact up to 131k distinct colours and folds to a
6-bit/channel grid above that, which bounds median cut's input on sources
with millions of distinct colours. Real footage crosses that line more
often than it looks like it should: three of the four cartoon clips in
our production corpus fold, because they arrive as decoded video and
carry the codec's noise, not as clean cel art. Synthetic and few-colour
sources stay exact.

## Prefilters: `--hold` and `--smooth`

Compressed-video input carries a few LSB of noise on every pixel of every
frame, and on flat content that noise — not the picture — decides which
of two neighbouring palette entries a pixel lands on. Re-rolled each frame
it turns static fills into per-frame index churn that defeats the delta
encoder. Two optional prefilters, both running as pipeline stages between
the reader and the histogram pass, address this at the source:

- `--hold N` keeps a pixel at its held value while the input stays within
  the hold window of a per-pixel running mean (and within 1.5x that of the
  held value, which bounds the lag on slow drifts). The window is not N
  itself: it adapts per frame to the measured frame-to-frame noise — 2.5x
  its median change, floor 4 — and N is the cap, so a clean source settles
  around 4-5 whatever cap you set and only a grainy one uses the whole
  budget. Static regions become byte-identical across frames and drop out
  of the delta entirely.
- `--smooth N` replaces each pixel with the mean of the 5x5 neighbours
  within N of it, so grain averages out inside fills while nothing across
  an edge is touched. With the grain gone, far fewer pixels escape the
  hold window, and the palette sees clean fills.

On a grainy 480x360 cartoon clip at `--lossy 30`: 9.10 MB baseline,
6.40 MB with `--hold 8`, 5.59 MB with `--smooth 24 --hold 8`, at equal
PSNR against the source; on a corpus of random cartoon clips the pair is
worth 8-18% with PSNR unchanged or up. Both are cheap: the hold runs on
one thread, the filter on a small pool, and the encode is faster overall
because the delta stage has less to do. They work on packed RGBA: y4m
input is converted once in the pool when either is on (frames that turn
out opaque drop to packed RGB after the first pass, as RGBA input does),
so a y4m clip with a prefilter holds about twice the resident set of one
without.

## Design

Every heavy stage is parallel under rayon — across frames, and for the
histogram across row strips within a frame as well:

1. **Read + histogram, overlapped.** A reader thread streams frames into a
   bounded channel while workers RLE-scan them into packed runs and route
   those runs into 256 buckets by red byte; a second pass then accumulates
   one bucket per task into an open-addressed table keyed by 24-bit color.
   Bucketing rather than giving each worker its own table is what keeps the
   merge free — bucket order *is* sorted order, so there is no duplicate
   per-worker state to dedup afterwards. Above 131k distinct colors the
   histogram folds to a 6-bit/channel grid, which is the common path on
   real footage; below 255 the palette is exact and the whole search is
   skipped. Scanning is split across 32-row strips as well as across
   frames, because a batch is only as deep as the reader has queued and
   that measured 12-15 frames against 40 workers. RGBA rows are hashed in
   place, straight out of the frame; only y4m pays a conversion into a
   scratch row. This pass also records, per frame, whether any pixel is
   transparent — the pipeline needs that before quantization to pick a
   disposal mode, and it doubles as the test for packing an all-opaque
   frame down to 3 bytes/pixel for the rest of the run.
2. **Palette: variance median cut in OkLab**, mirroring ffmpeg ≥5.x
   palettegen: the box with the largest single-channel squared error (in
   Lab) splits at its count-weighted median along that channel; each box
   emits the Lab average of its colors. Boxes that collapse to (or are
   ≥99% dominated by) one exact color emit that color byte-exactly.
   Sources with ≤255 distinct colors skip straight to a lossless palette.
3. **Exact nearest-color lookup** via per-cell candidate lists over a
   6-bit/channel RGB grid (locally sorted search): a triangle-inequality
   bound makes the argmin over the cell's candidates the true OkLab
   nearest for every color in the cell. Single-candidate cells answer from
   one table load, but on real footage 90-98% of *pixels* land in
   multi-candidate cells, so the hot path is really a probe of a
   per-thread direct-mapped memo cache, with the grid touched only on a
   miss. That cache is sized by dividing a fixed budget across the
   workers rather than fixing a per-worker size: what decides whether a
   probe is cheap is the sum of the workers' tables against the L3 they
   share, and past that point a cache *hit* costs a DRAM round trip. The
   build's 262k-cell × 256-color distance sweep runs on fearless_simd's
   portable f32 lanes.

Two hot paths are vectorized with fearless_simd behind runtime dispatch
(SSE4.2/AVX2/AVX-512/NEON picked per machine, independent of the build's
`target-cpu`), both verified byte-identical to the scalar paths:
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
   error-bounded match search described above, which intersects a
   symbol's substitution candidates with the trie node's children as a
   bitmask before walking the list — most nodes share no candidate with
   their children at all, and almost all of the rest share exactly one.
6. **Mux.** Single sequential write of pre-encoded chunks.

Correctness is pinned by unit tests (LZW roundtrip incl. forced clears,
palette exactness, nearest-map-vs-brute-force in OkLab, y4m parsing, delay
accumulation) and an end-to-end test that decodes weft's output with an
independent minimal GIF decoder and compares canvases byte-for-byte.

Dependencies: rayon and fearless_simd; the musl static build adds
mimalloc.

## Optimisation history

[`docs/experiments.md`](docs/experiments.md) records optimisations that
were implemented, measured and **rejected**, with the numbers that killed
them — Wu's quantizer, several `--dither auto` gate reworks, half a dozen
histogram and cache changes — along with the literature behind them. Read
the relevant entry before trying one of these again: the goal is to start
from where the last attempt stopped rather than from zero. It also carries
the measurement discipline this benchmark needs (interleave A/B runs; the
`read+hist` noise floor is ~5%), which is worth reading before producing
any number of your own.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
