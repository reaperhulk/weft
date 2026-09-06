# Changelog

## 0.5.0

weft no longer depends on rayon. Every parallel stage now runs on a
work-sharing pool built for weft's own shape (`src/pool.rs`). **Output is
byte-identical to 0.4.3** — verified on fourteen clips under four flag
sets at nine worker counts — and no flag, default, palette, or
compression threshold changed. (#51)

- Every parallel region in weft is the same shape: *n* independent items,
  run to completion before the caller continues. The pool is built to that
  contract: one closure pointer published per region, an atomic cursor
  that workers claim item ranges from, dense worker indices (so per-worker
  scratch is an array lookup rather than a thread-local probe), and the
  submitting thread doing a share of the work. There are no deques, no
  per-region allocation, and no splitting.
- Joining a region is optional. A worker that the OS descheduled, or that
  is still waking from a park, misses the region instead of stalling it.
  This is what keeps region overhead flat as the worker count grows: an
  empty region costs ~3 µs at 22 and 44 workers.
- Idle workers spin briefly, then poll with `yield_now`, then park; a
  worker whose previous idle ended in a park skips the spin the next time
  (adaptive spinning, after filament). When a slow decoder leaves the pool
  idle between batches this is a 13% CPU reduction against rayon, and it
  is neutral when input is fast.
- `--threads N` now means N runnable threads, the submitting thread
  included. Under rayon it was N worker threads plus a blocked main
  thread, so the same N does the same amount of parallel work; `--threads
  1` now runs entirely inline with no thread spawned and no
  synchronization.
- The palette's canonicalizing sort is a parallel radix pass with a
  per-bucket comparison sort, preceded by a scan that skips the work when
  the entries already arrive sorted (the exact-histogram case). The
  result is exactly what `sort_unstable` produced.
- The musl static build no longer links rayon or crossbeam; the only
  remaining dependency is fearless_simd (plus mimalloc for musl).

Measured on a 22-core Broadwell Xeon at `--lossy 30 --dither auto --hold
12` over fourteen RGBA clips: geometric-mean throughput +3% at one
worker, +2-4% at 4-8, +5% at 16, +6% at 22, and +9% at 44 workers, with
5-20% less CPU time and 80-90% fewer context switches. See
[the measurement log](bench/pool.md) for per-clip results, hardware
counters, the region-latency microbenchmark, and validation.

Regression coverage adds sixteen pool unit tests (ordering and
one-run-per-item at every scheduling boundary, drop-exactly-once through
the by-value forms, panics from items and from per-worker init on both
the caller and worker paths, parked workers waking, publishes raced
against parking, late joiners backing out, concurrent submitters,
concurrent pools, and drop in every worker state), thread-invariance at 2
and 3 workers, and a many-colour clip that exercises the coarse-binned
histogram and the parallel sort across worker counts.

## 0.4.3

Performance improvements for production encoding, including smaller worker
pools. **Output remains byte-identical on the benchmark and regression
corpora** across the optimisation stages. No CLI flag, default, palette
quality, or compression threshold changed. (#49)

- Temporal hold processes disjoint pixel ranges on a separate pool of up
  to four workers, preserving frame order and the adaptive noise threshold.
  Small images and fewer than eight configured workers retain serial hold.
- Auto dithering skips a redundant pass on opaque rows with no active
  dither tiles when no activity scale is needed.
- Palette construction uses runtime-dispatched AVX-512F kernels for
  distances, nearest-colour selection, and candidate filtering on supported
  x86_64 CPUs. Short palettes retain the existing paths; the x86-64-v3
  build requirement and AVX2/NEON fallbacks are unchanged.
- Lossy LZW follows exact-only trie continuations in a loop, reducing
  recursive calls while preserving candidate order, visit budgets, error
  bounds, and match selection.
- Larger quantize/encode batches reduce underutilised tail batches on
  short clips. Batch size increases from four frames per worker to
  `max(8 * workers, min(16 * workers, 160))` frames. Buffer storage remains
  bounded independently of clip length, but larger batches and the
  additional hold workers can increase peak memory.

The recorded 14-clip RGBA benchmarks use `--lossy 30 --dither auto --hold 12`.
The first optimisation stage improved geometric-mean throughput by 14.0%
on a Xeon E5-2696 v4 at 22 workers. On Cascade Lake, the work through the
AVX-512 stage measured 5.7% over 0.4.2 at 40 workers; the subsequent LZW and
small-pool batching changes added 7.0%, 2.9%, and 1.3% at 10, 20, and 40
workers respectively, relative to the AVX-512 stage. These are separate
comparisons on specific hosts and corpora, not a universal speedup. See
[the benchmark log](bench/hill-climb.md) for per-clip results, memory
measurements, and validation.

Regression coverage now includes production flags at 10, 20, and 40
workers, duplicate frames spanning batch boundaries, skipped dither rows,
AVX-512 equivalence, and exact-only LZW traversal.

## 0.4.2

Performance and one build-requirement change. **Output is byte-identical
to 0.4.1**, verified on four 480×360 RGBA clips at every thread count
measured. No flag or default changed.

- **x86_64 binaries now require an x86-64-v3 CPU** (AVX2/FMA/BMI2: Intel
  Haswell 2013+, AMD Zen 2017+); older machines fail at startup with an
  illegal-instruction error. `.cargo/config.toml` sets the target level
  for x86_64 builds, and `RUSTFLAGS="-C target-cpu=x86-64"` overrides it
  for a baseline build. aarch64 is unchanged.
- The fast OkLab cube roots in the nearest-colour scan are computed as
  one explicit 4-lane vector with a safe padding lane. Left to the
  compiler, an AVX build packed them with a zero pad lane that went
  subnormal on every call, and each subnormal operand is a ~150-cycle
  microcode assist — which is why `target-cpu=x86-64-v3` used to measure
  up to 1.6× slower on the quantize phase. With the pad lane fixed the
  v3 build wins: −3% to −4% end to end on 480×360 RGBA at 8 and 40
  threads (the explicit vector alone is −2% to −3% on a baseline build).
  A test proves the packed and scalar paths bit-identical over every one
  of the 2²⁴ sRGB colours.

Measured and rejected on the same workload (see the Cargo.toml notes):
`x86-64-v4` and `native` builds are 2–4% slower than v3 on Cascade Lake,
and mimalloc on the glibc build is 40% slower at 40 threads.

## 0.4.1

Performance and documentation. **Output is byte-identical to 0.4.0**,
verified across 100 encodes (four clips × five thread counts × five flag
combinations). No flag or default changed.

End to end on 480×360 RGBA over a pipe at 40 threads: −17% on a 41-frame
clip, −12% on 74 frames, −6% on 137 and 147 frames. The gains are largest
on short clips and on piped RGBA; the synthetic y4m clips in `bench/`
barely move.

- Lossy LZW intersects a symbol's substitution candidates with the
  dictionary node's children as a bitmask instead of testing them one at a
  time — 94% of those tests were failing. LZW phase 0.68–0.80×. (#40)
- The nearest-colour memo cache is sized by dividing a budget across
  workers instead of a fixed 1 MiB each, which had put 20 MiB per socket
  against 16 MiB of L3 — past that point a cache *hit* costs a DRAM round
  trip. Quantize phase 0.56–0.85×. (#42)
- Pass 1's histogram scan splits each frame into 32-row strips, so it is
  no longer bounded by how many frames the reader has queued (measured at
  12–15 against 40 workers). read+hist 0.85–0.97×. (#43)

Fixed:

- A misplaced `#[test]` attribute had silently disabled the lossy LZW
  size/error-bound test. (#41)
- `Cargo.lock` carried a stale version, so every build regenerated it.
  (#40)

Documentation:

- New `docs/experiments.md`: optimisations that were implemented,
  measured and rejected, with the numbers that killed them — Wu's
  quantizer, four `--dither auto` gate reworks, and more. (#44)
- README corrected against the code and every benchmark table re-measured.
  Three claims were wrong: the per-pixel lossy cap has applied to
  `--dither none` only since 0.4.0, the histogram is bucket-routed rather
  than per-thread, and `--hold N` caps an adaptive window rather than
  being a fixed threshold. (#45)

## 0.4.0

`--dither auto` became the default, replacing `bluenoise`; output differs
from 0.3.0 for the same input.

- New `--hold N` (temporal hold, adaptive window capped by N) and
  `--smooth N` (edge-preserving 5×5 grain filter), both off by default.
- New `--dither auto`: blue noise only in 32×32 tiles whose nearest-colour
  map shows banding contours.
- Palette gains three Lloyd refinement passes.
- The lossy error cap is scaled per pixel under `--dither none`, and
  deliberately left flat under `auto` — scaling it there removed the LZW
  noise that had been masking contours the gate missed.
- Much finer `--stats` output.

Releases before 0.4.0 predate this file; see the git history.
