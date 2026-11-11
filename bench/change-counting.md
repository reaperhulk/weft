# Counting by change, and four smaller things — 2026-11-11

Baseline: main `fc025ea` (weft 0.5.0 plus dependency bumps). This round
makes the encoder **35% faster** at `--lossy 30 --hold 12` on four
threads, for 28% less CPU time, with byte-identical GIFs. Every clip in
the corpus improves.

**Read the host line before the numbers.** This was measured on a 4-vCPU
VM. The earlier rounds in this directory ran on 20-40 cores, where the
read-and-histogram phase is paced by the reader rather than by CPU; how
much of this gain survives there is not known, and the notes below say
which parts should and which may not.

## Measurement

Host: 4 vCPUs of an Intel Xeon at 2.1 GHz with AVX-512 (KVM guest), 15
GiB, Linux 6.18, Ubuntu 24.04, glibc build. Rust 1.98.1 (CI's pinned
toolchain) for the headline tables, repository release profile and
`x86-64-v3`. No hardware counters are exposed to this guest, so there are
no cycle or instruction figures; profiles are `perf record -e cpu-clock`.

Corpus: the 20 Frinkiac excerpts of `fetch_frinkiac.py`, re-fetched for
this round: 2,880 frames, fifteen clips at 480x360 and five at 480x270,
predecoded to RGBA. The service re-renders on request, so the bytes are
not those of [frinkiac-manifest.json](frinkiac-manifest.json) even though
the windows are; [change-counting-manifest.json](change-counting-manifest.json)
records what was measured.

Each figure is the median complete process wall time: startup, the
cached input read, encoding, the GIF write, teardown. Binaries alternate
within every repetition and the order within each pair is shuffled, with
one warm-up per binary and clip. Tables give the geometric mean of
per-clip median ratios. Output hashes are compared on every clip.

| Configuration | Runs | Baseline | Final | Wall | CPU |
|---|---:|---:|---:|---:|---:|
| `--lossy 30 --hold 12`, 4 threads | 15 | 189.6 ms | 140.7 ms | **0.742x** | 0.722x |
| defaults, 4 threads | 9 | 144.6 ms | 108.4 ms | **0.750x** | 0.721x |
| `--lossy 30 --hold 12`, 4 threads, piped in with `cat` | 9 | 208.0 ms | 150.0 ms | **0.721x** | 0.713x |
| `--lossy 30 --hold 12`, 2 threads | 7 | 319.7 ms | 229.2 ms | 0.717x | 0.726x |
| `--lossy 30 --hold 12`, 1 thread¹ | 5 | 653.4 ms | 485.3 ms | 0.743x | 0.759x |
| defaults, 1 thread¹ | 3 | 496.5 ms | 382.2 ms | 0.770x | 0.772x |

¹ Rust 1.97.0 builds, and taken before the last commit of the round
(which only affects one thread, and only favourably).

Production flags, 4 threads, 15 runs:

| Clip | Size | Frames | Baseline ms | Final ms | Throughput gain |
|---|---:|---:|---:|---:|---:|
| 01-S01E01-2000ms | 480x360 | 48 | 69.3 | 50.8 | +36.4% |
| 02-S02E04-2500ms | 480x360 | 60 | 109.6 | 85.7 | +27.8% |
| 03-S03E08-3000ms | 480x360 | 72 | 131.0 | 100.3 | +30.7% |
| 04-S04E12-3500ms | 480x360 | 84 | 148.5 | 108.1 | +37.4% |
| 05-S05E02-4000ms | 480x360 | 96 | 154.3 | 104.6 | +47.6% |
| 06-S06E06-4500ms | 480x360 | 108 | 269.9 | 247.0 | +9.3% |
| 07-S07E10-5000ms | 480x360 | 120 | 135.5 | 77.8 | +74.2% |
| 08-S08E14-5500ms | 480x360 | 132 | 262.6 | 207.7 | +26.4% |
| 09-S09E03-6000ms | 480x360 | 144 | 194.1 | 127.8 | +51.8% |
| 10-S10E07-6500ms | 480x360 | 156 | 204.9 | 131.8 | +55.5% |
| 11-S11E11-7000ms | 480x360 | 168 | 298.2 | 236.3 | +26.2% |
| 12-S12E15-7500ms | 480x360 | 180 | 254.4 | 188.9 | +34.7% |
| 13-S14E02-8000ms | 480x360 | 192 | 240.0 | 155.0 | +54.8% |
| 14-S16E06-8500ms | 480x360 | 204 | 260.0 | 194.3 | +33.8% |
| 15-S18E10-9000ms | 480x360 | 216 | 511.5 | 452.2 | +13.1% |
| 16-S20E14-9500ms | 480x270 | 228 | 422.2 | 378.9 | +11.4% |
| 17-S23E03-9900ms | 480x270 | 237 | 200.7 | 152.5 | +31.6% |
| 18-S27E07-2200ms | 480x270 | 52 | 71.8 | 58.0 | +23.8% |
| 19-S31E11-6200ms | 480x270 | 148 | 145.3 | 107.7 | +34.9% |
| 20-S35E15-9800ms | 480x270 | 235 | 193.7 | 127.5 | +51.9% |

The clips that gain least (06, 15, 16) are the ones where most pixels
change every frame; the ones that gain most are the stillest.

y4m at the defaults (Rust 1.97.0 builds, 7 runs; the first six are corpus
clips re-encoded to yuv420p, the rest are `gen_inputs.sh`'s synthetic
clips, which the README's table is built on):

| Clip | 4 threads | 1 thread |
|---|---:|---:|
| 01-S01E01 | 0.736x | 0.738x |
| 04-S04E12 | 0.765x | 0.909x |
| 07-S07E10 | 0.573x | 0.649x |
| 11-S11E11 | 0.777x | 0.907x |
| 15-S18E10 | 0.872x | 0.968x |
| 20-S35E15 | 0.589x | 0.692x |
| testsrc | 0.760x | 0.658x |
| mandel | 0.989x | 1.067x → 1.026x² |
| gradients | 0.873x | 0.809x |
| life | 0.806x | 0.778x |
| static | 0.966x | 0.820x |

² The one regression found anywhere: a mandelbrot zoom at `--threads 1`,
where every pixel changes into as many runs as pixels. The second figure
is after the last commit of the round.

The static musl build (mimalloc) measures the same: 0.796x wall, 0.767x
CPU at production flags and 4 threads, five runs (Rust 1.97.0, on a
noisier stretch of the host than the headline table).

## Retained changes

Cumulative wall-time ratios at production flags and 4 threads, as each
landed (separate 5-11 run comparisons, so they do not multiply out
exactly):

| Change | Step | Cumulative |
|---|---:|---:|
| LZW: skip the walk through transparent runs | 0.92x | 0.92x |
| Histogram: count by change, RGBA | 0.86x | 0.80x |
| Histogram: y4m too, per-frame fallback | 1.00x on RGBA | 0.79x |
| 1 MiB stdin pipe (piped input only) | 0.945x | — |
| Zeroed tables fault once per page | 0.94x | 0.74x |

**LZW: transparent runs.** With `--hold`, 85% of the pixels handed to LZW
on a typical clip are the transparent index, in long runs. A match
starting in one walked the dictionary's chain of all-transparent strings
a pixel at a time: one hash probe each, plus the lossy search's
visit-budget, child-bitmap and best-match bookkeeping, all to reach a node
determined by the run's length alone. The encoder now records that chain
as it builds it (`RepChain`) and starts the match at the deepest node the
run reaches, found by a word-wise scan. On clip 12, 94% of transparent
pixels are covered by the skip. Nothing about the search changes: same
node, same position, same remaining visit budget — a test compares all
three, including searches that exhaust the budget. Applies to lossless
encoding too (0.945x at the defaults).

What is left of the lossy search is 3.2 visited nodes per emitted code at
about 65 cycles each, on tables that sit in L2. That is the search doing
what it is for; no further structural win was found there.

**Histogram: counting by change.** See the section comment in
`palette.rs` and the README's design notes. The pass's cost was the scan
itself — a scalar run-length loop with a mispredicted branch per run, at
a run every 1.5-2.7 pixels — not the tables behind it. Comparing 64 bytes
of a frame with the previous frame's is nearly free, and on this corpus
only 5-37% of pixels differ with `--hold 12` (10-62% without). On clip
12 at one thread the phase goes from 290 ms to 137 ms.

The per-frame scan is kept for the content where it wins: smooth frames
in which everything changes (a fade across flat fills: change counting
+33% on the phase, +6% end to end, before the fallback; +12% and +2%
after). Notably it does *not* win on noisy or detailed full-change
content (noise, a mandelbrot zoom), where the change counter's
bit-mask loop beats the scan's branch per run even at one entry per
pixel — which is why the switch asks for three times more change entries
than runs rather than merely more.

**stdin pipe.** `F_SETPIPE_SZ` to 1 MiB. The reader's own read time falls
by about a third (70 → 46 ms on clip 12); at four threads that is 5% of
wall time. On a machine where the reader paces the phase it should be
worth more, and when the decoder is the bottleneck, less.

**Page faults.** At four threads 5.1% of all samples were in TLB
shootdowns, from zero-initialised tables whose pages were first touched
by a read: the kernel maps its shared zero page, and the first write
after that is a second fault that must interrupt every other core. The
coarse bins and the shared nearest-colour cache were the two that
mattered. One thing worth knowing if you touch this: the reader's
existing "allocate, then `resize` with zeros" idiom for forcing real
writes is folded back into `calloc` by LLVM in at least one place (the
bins; confirmed in the disassembly). `zeroed_by_writing` makes the
pointer escape first. The interrupts fan out to every core running the
process, so this is the change most likely to be worth *more* at 40
threads than it was here.

## Cost

Peak RSS, median of five, production flags, 4 threads: 79.6 → 87.9, 124.2
→ 130.1 and 115.7 → 120.7 MiB on three RGBA clips; 66.5 → 70.9 and 56.1 →
55.3 MiB on two y4m clips at the defaults. The additions are 4 bytes per
pixel of `since` state, one retained RGBA frame (or 4 bytes per pixel of
colour keys for y4m), and the two tables above now being resident in
full rather than in the pages touched.

## Rejected

- **Returning input buffers to the reader** once packed to RGB, instead
  of allocating and zeroing one per frame: 1.5-2% *slower*, file-fed and
  piped. glibc was already handing those pages to the packed frames, so
  recycling them only moved the faults there. (Second time this has been
  tried; the first was on macOS and was neutral.)
- **Aborting the second encode** of a frame that is encoded both punched
  and opaque as soon as it exceeds the first. Byte-identical in
  principle, but only 1-5% of frames on this corpus are encoded twice,
  so it was not built.
- **Counting by change with no fallback.** Simpler, and within noise of
  the adaptive version on real clips, but +6-10% end to end on
  flat-colour fades.
- **Switching per frame on change count against sampled run count**, the
  first adaptive rule. Slower than no fallback at all: strips flapped on
  every scene cut, paying the hand-over for one scanned frame, and the
  per-frame scan was chosen for busy strips it is slower on. The rule
  that works needs two consecutive frames and a 3x margin.
- **Sampling runs in every fourth block**, changed or not: 10% of the
  phase. Sampling only inside blocks that already changed costs nothing
  measurable.

## Not done

Quantization still recomputes every pixel of every frame, and the same
observation applies: the nearest-colour lookup alone is about 6% of CPU
and is a pure function of a source pixel that usually has not changed.
Reusing it needs each worker to take consecutive frames, which at 40
threads and 96 frames leaves runs of two or three; it looked like 2-5%
for a scheduling change, and was left.

Fresh pages for the packed frames are another 3.5% of samples at four
threads. Transparent huge pages would remove most of it but need the
frames allocated outside `Vec`.

## Validation

- `cargo fmt --check`, `cargo clippy --release --all-targets -- -D
  warnings` and `cargo test --release` on Rust 1.98.1 (98 unit, 13
  integration); clippy also clean for `aarch64-unknown-linux-gnu`, which
  compiles the NEON paths but was not run. The musl target builds and
  produces the same bytes.
- **960-run identity sweep** against the baseline binary: seven corpus
  clips, eight synthetic RGBA clips (noise, fades, a zoom, static bars,
  two with transparency crossing the threshold, 3x2 and 37x23 frames) and
  five y4m clips, under six flag sets (defaults; `--lossy 30 --hold 12`;
  `--dither none --lossy 30`; `--colors 64 --dither bluenoise --smooth
  16`; `--lossy 80 --dither sierra2`; `--dither bayer --hold 8 --smooth
  24`) at 1, 2, 3, 4, 8 and 13 threads. 117 distinct digests, no
  mismatch.
- New tests: the LZW skip against the pixel-at-a-time walk over 60
  generated frames with and without lossy matching and a scale map; the
  search state (best match and visits left) with and without the skip,
  including exhausted budgets; and the strip counters against a
  per-frame scan over 56 generated clips that drive both modes and both
  hand-overs, for RGBA and planar input. Each was checked to fail under
  mutation (13 of 13 for the counters).
