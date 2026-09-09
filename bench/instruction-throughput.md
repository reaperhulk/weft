# Instruction throughput on production GIF encodes

## Revision after x86-64 production testing

The original cross-platform revision `1ecafcd` regressed x86-64. External
production testing supplied by the user found the following candidate/main
ratios on 12 clips (21 paired reps at 20 threads, 11 at 8 threads):

| Threads | Wall | CPU | Instructions | Clips with instruction regressions |
|---|---:|---:|---:|---:|
| 20 | 1.021× | 1.032× | 1.043× | 12/12 |
| 8 | 1.013× | 1.022× | 1.046× | 12/12 |

Quantization grew 11.1% and LZW 2.6%, despite a 21.5% nearest-map improvement.
The NEON kernel is not compiled on x86; its fallback is the original distance
kernel. The cache permutation changes both lookup and prefetch address
calculations, so the map-build win cannot be separated from the quantization
regression merely by removing NEON. Exact component attribution still needs
x86 ablation; the stage timings alone do not establish it.

The revised PR enables tiled cache addressing and deferred LZW stores only
on Apple ARM64. Other targets retain baseline RGB-major addressing,
prewarming, prefetch addresses and LZW bookkeeping. The NEON distance kernel
remains ARM64-only. No x86 performance improvement is claimed. A server
retest is still needed to verify the revised binary has no regression.

A follow-up M5 run of the restricted implementation (14 clips, 11 paired
reps, 18 threads) retained the gains: wall −3.4%, instructions −2.4%,
cycles −4.5%, IPC +2.2%. Every GIF was byte-identical to main.

The results below describe the original M5 measurements, not x86 gains.

Against main `d09af78`, using `--lossy 30 --hold 12 --dither auto`.
Apple M5 Max, 18 cores (6 Super + 12 Performance), macOS 26.6.2;
Rust 1.96.0 / LLVM 22.1.2, unchanged release profile. No compiler tuning or PGO.

The full process uses **2.4% fewer instructions, 4.3% fewer cycles and
4.2% less CPU time**, with **1.9% higher IPC and 3.2% less wall time** at
18 worker threads. These are modest, measured gains, not a large breakthrough.

## Changes

- Store each 4×4×4 RGB cell contiguously in the existing 16 MiB exact
  source-color cache. Reuse precomputed grid keys in staged lookups and
  write the 64 classified results contiguously when prewarming. This is
  a permutation of the cache, with no approximation or palette change.
- Defer best-match stores in lossy LZW until an exact chain ends. Each
  intermediate match has the same error and a strictly shorter length.
  Traversal order, tie-breaking and visit-budget consumption are preserved.
- On ARM64, compute per-pixel RGB/RGBA absolute distances with NEON byte
  absolute differences and pairwise widening sums, replacing channel
  shifts and masks. Other architectures retain their existing kernels.

The first two changes were originally applied across architectures; they are
now enabled only on Apple ARM64. An 11-pair M5 ablation on all 14 clips
measured 1.7% less wall time with just these changes; adding NEON brought
that run to 3.3% less wall time. This is an M5 measurement of portable
code, **not evidence of an x86 speedup**.

## Measurement

Four development excerpts and five previously random excerpts, plus five
new six-second excerpts selected through Frinkiac's random endpoint:
S07E21, S18E13, S24E07, S20E08 and S12E14. All 14 require quantization.
The new clips were first measured after selecting the retained changes;
results also held in the subsequent final-build run below.

25 paired runs per clip at 18 threads; one warmup per binary/clip, shuffled
pair order and shuffled binary order within each pair. Values are geometric
means of per-clip medians. Lower ratios are better except IPC. The complete
weft process is measured, including startup, input reads and GIF writes;
MP4 decoding is excluded because all inputs were decoded beforehand.

`macos_counters.c` reads `proc_pid_rusage(..., RUSAGE_INFO_V6, ...)` after
`waitid(..., WNOWAIT)` reports child exit and before reaping it. Instructions
and cycles are actual process hardware counters, not timing estimates.
CPU time comes from `wait4`. IPC is instructions/cycles for each observation.
No root or Instruments is needed on this machine. The helper fails when
instruction/cycle counts are unavailable. Arbitrary cache-miss/branch PMU
events were not available through this measurement path.

| Cohort | Wall | CPU | Instructions | Cycles | IPC |
|---|---:|---:|---:|---:|---:|
| 9 development clips | 0.9701× | 0.9594× | 0.9747× | 0.9594× | 1.0161× |
| 5 fresh random clips | 0.9642× | 0.9548× | 0.9778× | 0.9536× | 1.0251× |
| All 14 clips | 0.9680× | 0.9578× | 0.9758× | 0.9573× | 1.0193× |

Aggregate IPC can change with work distribution across core classes. The
kernel's `ri_pinstructions / ri_pcycles` subset also improves: 0.8% across
all clips and 1.1% on fresh clips. Raw records retain both total and subset
counts; the machine has Super and Performance cores, not efficiency cores.

| Clip | Wall ratio | Instructions ratio | IPC ratio |
|---|---:|---:|---:|
| S01E01-2000 | 0.9967× | 0.9716× | 0.9891× |
| S08E14-5500 | 0.9755× | 0.9836× | 1.0123× |
| S18E10-9000 | 0.9751× | 0.9948× | 1.0211× |
| S35E15-9800 | 0.9526× | 0.9703× | 1.0223× |
| 01-S14E15-1030699 | 0.9783× | 0.9641× | 1.0157× |
| 02-S06E08-1064692 | 0.9696× | 0.9612× | 1.0098× |
| 03-S20E10-984820 | 0.9456× | 0.9683× | 1.0573× |
| 04-S21E09-433436 | 0.9852× | 0.9880× | 1.0066× |
| 05-S19E13-1001504 | 0.9535× | 0.9707× | 1.0115× |
| 01-S07E21-638096 | 0.9570× | 0.9701× | 1.0172× |
| 02-S18E13-596596 | 0.9710× | 0.9750× | 1.0230× |
| 03-S24E07-932348 | 0.9614× | 0.9793× | 1.0409× |
| 04-S20E08-305680 | 0.9632× | 0.9790× | 1.0286× |
| 05-S12E14-195779 | 0.9682× | 0.9858× | 1.0160× |

## Limits and concurrency

With `--threads 1`, 11 paired runs per clip: wall time was essentially
unchanged (0.2% lower), CPU time 2.3% lower, instructions 2.5% lower and IPC
0.2% lower. The IPC improvement is **not universal across thread counts**.

Nine paired batches of 24 encodes, 18 threads per encoder, fixed identical
clip order for each binary (the first ten clips occur twice per batch):

| Concurrent encoders | Main ms/encode | Candidate ms/encode | Ratio |
|---|---:|---:|---:|
| 1 | 44.610 | 43.468 | 0.9744× |
| 10 | 28.731 | 28.172 | 0.9805× |

Native x86-64 performance has not been measured. Native Linux CI validates
correctness/build compatibility only. These changes are based on main and
do not include the separate hold-histogram banking PR #58.

## Correctness

All measured GIFs are byte-identical to main, including the fresh corpus,
one-worker runs and concurrent batches. A separate 30-case check compares
YUV420/422/444 input, five dither modes, and 1/18 threads with production
hold/lossy settings. Thus no before/after visual approval is needed for
these changes.

Release tests and all-target Clippy pass. The added randomized LZW test
compares best match, error, early termination and remaining visit budget
against a simple recursive reference over 3,200 cases, including transparent
symbols and position-dependent error caps. Existing source-cache tests check
cold/warm/coarse results and racing publications against a reference map.

## Reproduce

`counter-manifest.json` records source requests and input checksums. Place
each decoded `name.rgba` beside a copy of the manifest. Keep main and PR
binaries from the same release profile/toolchain. Run:

```sh
cc -O2 -Wall -Wextra -Werror bench/macos_counters.c -o /tmp/weft-counters
python3 bench/counter_bench.py \
  --helper /tmp/weft-counters --manifest /path/to/data/manifest.json \
  --binary main=/path/to/main/weft --binary candidate=/path/to/pr/weft \
  --threads 18 --runs 25 --output /tmp/weft-counter-results.json
```

The runner verifies input checksums and output identity, records every
observation and binary hash, and prints per-clip-median aggregates.
`instruction-throughput-results.json` contains the aggregate and per-clip
results reported here. Raw observations are also supplied in the task's
`weft-instruction-throughput-evidence.zip`.

Larger dictionaries, dense child tables, maintained chain/depth metadata,
precomputed alternate masks, and an explicit DFS stack were screened and
discarded when they added cycles or failed to improve elapsed time. A
zero-dither specialization cut more instructions but did not meet the IPC
target; it is not included.
