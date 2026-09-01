# Changelog

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
