# Experiment log

Things that were tried, measured, and **rejected**, with the numbers that
killed them. The point of this file is that optimising weft has a long tail
of ideas that sound good, and several of them have now been implemented
twice. If you are about to try something here, read the entry first — not
to be told "no", but so you start from where the last attempt stopped
rather than from zero.

Entries record what was measured, not what was expected. Where a
hypothesis was wrong, the wrong hypothesis is kept.

All numbers are from the `cghmc-bench` harness (480×360 RGBA fed over a
pipe — the shape production runs, which differs from `bench/` on four axes
that turn out to matter) on a 2×20-core Xeon Gold 6248, 16 MiB L3 per
socket, `--lossy 30 --hold 12 --dither auto` at 40 threads unless stated.

That harness lives outside this repo and is not checked in; it holds four
real clips as raw RGBA plus `sweep.py` (timing and `--stats` breakdown)
and `cambi.py` (banding score). Fixture names below — `short-41f`,
`banding-74f`, `busy-137f`, `heavy-147f` — refer to it. If you do not have
it, the numbers here are still the record of what was tried; reproducing
them needs the harness.

## Measurement discipline

Read this before trusting any number you produce.

- **Interleave A/B runs.** Alternate binaries within each repetition.
  `cghmc-bench/sweep.py` runs all reps of A and then all of B, and that
  has produced false readings on this box more than once — a change that
  looked like +3.6% and another that looked like −16% were both drift.
- **`read+hist` noise is ~5% run to run.** Any single-pass comparison of
  that phase smaller than that is not a result. Phase-level comparisons
  want a median of 9–15 interleaved runs.
- **Gate commits on CI's pinned toolchain (1.98), not the local one.**
  Local clippy is five releases behind and silently misses lints; see
  `.github/workflows/ci.yml`.
- **Perf changes should be byte-identical.** Most of the wins below were
  verified by md5 across fixtures × thread counts × flag combinations.
  A change that alters output needs CAMBI, not just PSNR — see below.
- **PSNR and CAMBI disagree here, and CAMBI is the one that matches what
  you see.** A change that raised PSNR while making banding materially
  worse has already shipped once and been reverted. Compare CAMBI only at
  equal `--lossy`: lossy LZW noise breaks up contours and lowers it.

## Palette

### Wu's variance-minimising quantizer — rejected

Implemented, measured, abandoned. Code preserved on
`experiment/wu-quantizer` behind a `WEFT_WU` environment variable (branch
kept only as a recovery point; it is not on a path to merge).

The implementation departs from the paper twice, both deliberately: the
moment grid is laid out in **OkLab** rather than sRGB so the cut planes
and the minimised variance are perceptual, matching `median_cut`; and the
moments accumulate the exact OkLab coordinates of the colours in each
cell, so a box's mean is the true weighted mean and only the *cut planes*
are quantised to the grid.

It is genuinely faster and genuinely worse:

| | result |
|---|---|
| `median_cut` phase | **0.39–0.62×** |
| whole palette phase | 0.72–0.85× |
| end-to-end | −4.4% to −11% — *but only after retuning the dither gate* |
| output size | −1.7% to −6.3% |
| CAMBI | **worse on 3 of 4 clips, in every configuration tried** |

At the stock gate the end-to-end numbers are 1.020 / 0.910 / 0.918 / 0.979
— a *regression* on short-41f — because Wu doubles the dithered-tile count
and the dithering eats the palette gain. See the gate section below.

Best case found (`RUN=16`, `DE_LO=0.008`) is CAMBI 0.358 on heavy-147f
against median cut's 0.193, and banding-74f's max never drops below 1.2 in
any Wu configuration. short-41f is the one clear exception, where Wu is
better (0.207 vs 0.456).

**Why**: Wu minimises squared error, and banding is not a squared-error
phenomenon. Grid-aligned boxes cannot allocate entries inside a smooth
gradient as finely as median cut working on exact colours, and a
low-variance gradient is exactly what Wu's objective deprioritises.

Two follow-ups ran to separate the cause, both negative:

- **6-bit grid** (`WU_N = 64`), to test whether coarse cut planes were the
  cause. Only half-recovers banding-74f (cmax 1.431 → 1.122, still worse
  than median cut's 0.747), makes CAMBI *mean* worse on both clips tested,
  and costs `median_cut` 7.0 → 20.4 ms — no faster than what it replaces,
  which removes the entire reason to use it. So it is not just lattice
  resolution.
- **Wu's grid with median cut's count-weighted split rule** (`WEFT_WU=2`),
  to separate the objective from the machinery. Worse than either parent:
  banding-74f cmax 1.663, heavy-147f 0.836 mean / 1.145 max. So it is not
  the cut rule either.

## The `--dither auto` gate

### What the detector actually knows

Worth understanding before tuning it. `band_score_row` computes
`flat[x]` = "x lies inside a run of ≥ `RUN` (8) equal indices", and a
candidate contour is two flat plateaus meeting with different indices.
`BandGate::pairs` then asks whether the two palette colours differ by an
OkLab distance in `[DE_LO, DE_HI)`.

**There is no notion of a gradient anywhere in that.** A smooth ramp and a
texture boundary are indistinguishable to the detector; the entire
discriminating burden falls on `DE_LO`. That is why the gate is sensitive
to the palette it runs on — it is a proxy, and the proxy stops correlating
when the palette's spacing changes.

Concretely, Wu's more evenly spaced palette puts far more adjacent pairs
inside the window, roughly doubling the tile count at the same threshold:
banding-74f 10.8% → 23.8%, heavy-147f 9.2% → 16.8%, short-41f 9.9% →
18.5%. busy-137f is the control — its tile count did not move (14.3% →
14.2%) and it is the one clip with no `quantize` regression under Wu.

### Raising `DE_HI` — no effect

Hypothesis: Wu under-allocates entries in gradients, so its steps there
are *larger*, exceed `DE_HI = 0.05`, and get classified as edges and
skipped. **Wrong.** Sweeping 0.05 → 0.08 → 0.12 under Wu moves nothing:
heavy-147f cmax 0.769 / 0.768 / 0.764, banding-74f pinned at 1.369
throughout. The steps are not above the edge threshold.

### Lowering `DE_LO` alone — works, but unaffordable

It does improve CAMBI monotonically (heavy-147f 0.420 → 0.345 → 0.317 at
0.016 / 0.012 / 0.008; banding-74f 0.527 → 0.430 → 0.275, which beats
median cut). But because the detector cannot tell a ramp from texture, it
fires on everything: at `DE_LO = 0.008` that is **43.9%** of tiles on
heavy-147f (**+12.9%** size) and **76.9%** on banding-74f. It spends the
entire reason for using Wu and still does not reach median cut on
heavy-147f.

Note also that fewer dithered tiles is **not** monotonically faster —
median cut at `DE_LO = 0.016` dithers less and is *slower* on two of four
clips (1.035 short-41f, 1.050 heavy-147f), with LZW 7.1% slower on
short-41f. Undithered regions give longer index runs and the lossy DFS
matches deeper. Correlation observed; mechanism not traced.

### Plateau width (`RUN`) — the real discriminator, unshipped

Gradient plateaus are wide; texture plateaus are barely 8px. Raising
`RUN` 8 → 16 lets `DE_LO` drop far enough to catch subtle steps while
*reducing* the tile count:

| clip | config | tiles | size | CAMBI | cmax |
|---|---|---|---|---|---|
| heavy-147f | mcut / RUN8 / .012 | 9.2% | — | 0.193 | 0.311 |
| heavy-147f | wu / RUN8 / .008 | 43.9% | +12.9% | 0.317 | 0.453 |
| heavy-147f | wu / RUN16 / .008 | **5.9%** | **−3.1%** | 0.358 | 0.700 |
| banding-74f | wu / RUN8 / .008 | 76.9% | +4.4% | 0.275 | 1.460 |
| banding-74f | wu / RUN16 / .008 | **15.0%** | **−3.5%** | 0.370 | 1.333 |

`RUN` is a compile-time constant used in the SIMD kernel, and
`BAND_PAD` must be ≥ `RUN` (it is 16; raise it to 32 before trying
`RUN = 24`).

**This was not proposed as a new default, and here is why.** Carried over
to median cut, `RUN16 / .008` buys 0.3–2.0% size for a CAMBI regression on
three of four clips (heavy-147f 0.193 → 0.242, busy-137f max 0.655 →
0.924). That is the same trade that shipped once as r90 and was reverted.
The tuning was found by sweeping against *Wu's* palette; median cut's
`DE_LO = 0.012` was already tuned for it in r99, over 20 clips. A real
proposal needs a joint `RUN × DE_LO` sweep against median cut over the
20-clip set judged on CAMBI, plus timings, none of which exists yet. The
honest prior is that r99 landed near a local optimum and the win may not
be there.

### Monotone-staircase test — secondary, mixed

A gradient steps monotonically through the palette's lightness ordering;
texture alternates. Prototyped (not on any branch) as a palette lightness rank
plus a same-sign check against the previous contour in the row, within a
`MONO_GAP` window, applied in the sparse candidate loop in `dither.rs`.

It tightens the tile count further and helps short-41f (0.281 → 0.247) and
busy-137f (0.272 → 0.213, max 1.065 → 0.757), but *worsens* heavy-147f
(0.358 → 0.379) and banding-74f (0.370 → 0.496). `MONO_GAP = 64` is far
too loose — two contours 64px apart agree by chance about half the time in
texture; 24 and 16 behave identically. Only the horizontal direction was
implemented; vertical would need a per-column sign array.

## Read + histogram

`read+hist` is the largest phase (37–51% of the encode), and pass 1's
**phase A** — frames to bucket-sorted runs — is 69–77% of it.

**Landed** (#43): phase A parallelised over (frame, strip) pairs rather
than frames alone. Batches are only 12–15 frames against a cap of 80,
because the reader → prefilter → serial-hold chain and phase A sit in
equilibrium, so frame-level parallelism alone left ~15 tasks for 40
workers.

Rejected on the way:

- **Pooling the per-strip run buffers** behind a mutex to avoid ~144
  allocations per batch — *slower*. Lock contention beats malloc here.
- **Fusing the RGB narrowing into the strip scan**, so each strip narrows
  the rows it just read while they are in L1 — *slower*. The eagerly
  zero-filled per-frame output buffer costs more than the re-read saves.
- **Word-wise `rgba_to_rgb`** (repacking 4 pixels through `u32` instead of
  12 byte stores) — *neutral*, despite being 33% of phase A's summed CPU.
  That CPU is already overlapped.
- **Aggregating runs within a strip.** The runs are genuinely 2.0–4.2×
  redundant inside one 32-row strip (5.2k–9.2k runs over only 1.3k–4.5k
  distinct colours), and collapsing them with a small open-addressed
  colour→count table is byte-identical. But the benefit tracks redundancy
  and the crossover sits right at 2×: banding-74f (4.2×) gained 12%,
  busy-137f (2.0×) lost 3%. Table sizes from 2k to 16k slots all split the
  same way. A wash.
- **`with_min_len`** to coarsen tasks — hurts monotonically (1.05× /
  1.07× / 1.14× at 2 / 4 / 8). The phase wants the finest split available.

**Scaling note.** `read+hist` stops scaling above ~20 threads and is ~6%
*worse* at 40 than at 20 on heavy-147f (68.1 vs 72.3 ms, interleaved,
median of 11); on busy-137f the gap is ~2%, inside noise. Total time still
strongly favours 40 threads (147.5 vs 172.3), so this is not worth acting
on — but it means the phase is memory/interconnect-bound, not
compute-bound, at high core counts. A first pass at this over-read the
effect as much larger; it is not.

Content facts worth not re-deriving: mean horizontal run is **1.5–2.7 px**
(these are noisy video frames, not clean cel art, so run-coalescing does
not pay), and there are **24k–42k distinct colours per frame**.

## Quantize

**Landed** (#42): the nearest-colour memo cache is sized by dividing a
fixed 8 MiB budget across workers rather than being a fixed 1 MiB each.
What matters is the sum of the workers' tables against shared L3, not one
worker's footprint — past that point a cache *hit* costs a DRAM round
trip. The old constant put 20 MiB per socket at 40 threads and cost 33% of
the phase.

At the shipped size the memo hit rate is 63% on heavy-147f and busy-137f and
92.7% on banding-74f, with 18–23% of lookups doing a full `resolve_off`.
That is *fine* — the misses are cheap because the table is L2-resident,
which is the whole point.

Rejected:

- **2-way set-associative memo cache**, both with promote-on-hit and
  read-only hits — 1.02–1.10× *slower* in both forms. The extra branch on
  the miss path beats the hit-rate gain, and promotion dirties the line.
  Direct-mapped is right.

## LZW

**Landed** (#40): the lossy DFS intersects the symbol's candidate set with
the trie node's child set (a 256-bit mask per symbol ANDed with
`child_bits`) before walking the list. 94% of the old scan's iterations
were bit tests that failed.

The compression-side literature is a dead end **for this project's goals**
— every result there trades speed for size, and weft wants the opposite:

- **Non-greedy / flexible parsing.** Horspool 1995 is the canonical LZW
  result: ~8% smaller at K=3, saturating by K=3. **It is not GIF-legal** —
  the gain depends on adding the *greedy* string to the dictionary while
  emitting a shorter code, which needs his modified decompressor. He also
  measured the decoder-compatible variant (suppress the update): 40.9% vs
  greedy's 41.0%, i.e. nothing.
- **flexiGIF** is the GIF-legal version of that idea (one-step lookahead
  plus brute-forced dictionary-reset placement): **~2%**, at "several
  magnitudes slower … seconds or even minutes for a medium-sized GIF".
- **RDO / `J = D + λR` parsing** (Oodle Texture's framing) is the
  principled version of what the lossy DFS approximates. Same direction:
  buys compression by spending time.

weft's lossy DFS is already a non-greedy search and already gets 12–33%,
so the ~2% ceiling from lossless flexible parsing is not worth pursuing.

**Vectorised hash probing** (Polychroniou et al., SIGMOD 2015) applies in
principle to `probe()`, which is open-addressed linear probing over an
L1-resident table — but probes measured at only 0.92 per DFS node, so
that is not where the time is.

## References

- Xiaolin Wu, *Efficient Statistical Computations for Optimal Color
  Quantization*, Graphics Gems II, 1991.
- R. N. Horspool, *The Effect of Non-Greedy Parsing in Ziv-Lempel
  Compression Methods*, DCC 1995.
  <https://webhome.cs.uvic.ca/~nigelh/Publications/LZ-non-greedy.pdf>
- Stephan Brumme, *flexiGIF — lossless GIF/LZW optimization*.
  <https://create.stephan-brumme.com/flexigif-lossless-gif-lzw-optimization/>
- O. Polychroniou, A. Raghavan, K. A. Ross, *Rethinking SIMD Vectorization
  for In-Memory Databases*, SIGMOD 2015.
  <https://dl.acm.org/doi/10.1145/2723372.2747645>
- Kornel Lesiński, *Lossy GIF compression* — the algorithm weft's lossy
  DFS is a port of. <https://kornel.ski/lossygif>
- Charles Bloom, *Rate allocation in Oodle Texture* — the `J = D + λR`
  framing. <http://cbloomrants.blogspot.com/2021/02/rate-allocation-in-oodle-texture.html>
- CAMBI (libvmaf's banding detector), the metric that tracks
  posterization where PSNR/SSIM do not. Scored via the `cghmc-bench`
  harness; needs an ffmpeg built `--enable-libvmaf` (the
  `mwader/static-ffmpeg` image ships one).
