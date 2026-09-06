# Replacing Rayon with a purpose-built pool — 2026-09-05

Baseline: `55b642c` (weft 0.4.3), which uses Rayon 1.10. The candidate is the same tree with `src/pool.rs` in its place and every parallel region rewritten against it. No algorithm, flag, default, or threshold changed.

Host: Intel Xeon E5-2696 v4 (Broadwell, 22 exposed CPUs, no SMT) under KVM, Linux 7.0. Both binaries use the repository release profile and `x86-64-v3`. Flags: `--lossy 30 --dither auto --hold 12 --fps 24`, varying `--threads`.

Fourteen predecoded RGBA clips, 7 timed runs per binary per worker count, in a deterministically shuffled order so the two binaries stay adjacent in time. Each figure is the median complete process wall time, including startup, the cached input read, encoding, and the GIF write. Input is fed from a regular file on stdin, so no pipe capacity sits in the measurement path. These measurements describe this host and corpus, not a universal speedup.

**Every output SHA-256 is identical to baseline**, on all fourteen clips at every worker count measured here, and separately across a dedicated sweep of all fourteen clips under four flag sets — `--lossy 30 --dither auto --hold 12`, `--dither none`, `--colors 64 --dither bluenoise --smooth 16`, and `--lossy 80 --dither sierra2` — re-run after every change to the pool, at 1, 2, 3, 5, 7, 8, 13, 22 and 40 workers across the three sweeps: 2,016 runs, 56 distinct digests, no disagreement anywhere. Palette, dithering, compression decisions, frame timing and file size are unchanged.

## Throughput by worker count

| Workers | Geometric mean wall time, Rayon → pool | Throughput gain |
|---|---:|---:|
| 1 | 774.1 → 749.3 ms | **3.3%** |
| 4 | 297.6 → 291.7 ms | **2.0%** |
| 8 | 174.3 → 168.3 ms | **3.6%** |
| 16 | 124.5 → 118.3 ms | **5.2%** |
| 22 | 115.7 → 109.4 ms | **5.8%** |
| 44 | 126.6 → 116.3 ms | **8.9%** |

The curve is U-shaped, and both ends are the same cause: what the pool removes is per-region scheduling, and the number of regions is fixed by the clip. At one worker Rayon still hands every region to its single worker thread and blocks the caller, while the pool spawns nothing and runs the region inline — a handoff per region against none. In the middle both are cheap relative to the work. Past eight workers the cost of a dispatch grows with the number of threads that have to be woken and gathered for each of those fixed regions, and the gap opens again; at 44 workers on 22 cores the pool is also better behaved under oversubscription, where Rayon's sleep state machine spends most of its context switches.

## Per clip

| Clip | Size | 1w Rayon → pool ms | gain | 4w Rayon → pool ms | gain | 8w Rayon → pool ms | gain | 16w Rayon → pool ms | gain | 22w Rayon → pool ms | gain | 44w Rayon → pool ms | gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old | 480×360 | 1479.1 → 1448.5 | 2.1% | 374.7 → 364.1 | 2.9% | 234.5 → 231.0 | 1.5% | 159.0 → 146.4 | 8.6% | 147.6 → 134.8 | 9.5% | 159.4 → 147.4 | 8.2% |
| cel | 480×360 | 1521.5 → 1481.3 | 2.7% | 434.7 → 417.9 | 4.0% | 235.4 → 231.5 | 1.7% | 154.0 → 153.2 | 0.5% | 147.5 → 144.4 | 2.1% | 158.5 → 145.7 | 8.8% |
| modern | 480×270 | 681.0 → 686.3 | -0.8% | 278.4 → 275.5 | 1.1% | 162.0 → 158.2 | 2.4% | 120.2 → 115.7 | 3.9% | 117.0 → 107.0 | 9.4% | 126.5 → 116.4 | 8.7% |
| modern2 | 480×270 | 673.8 → 643.1 | 4.8% | 281.6 → 279.4 | 0.8% | 161.1 → 154.3 | 4.4% | 118.8 → 110.0 | 8.1% | 106.7 → 100.7 | 5.9% | 121.1 → 107.5 | 12.6% |
| grain | 480×360 | 844.3 → 806.7 | 4.7% | 341.2 → 337.2 | 1.2% | 196.0 → 190.2 | 3.0% | 134.7 → 132.9 | 1.3% | 128.0 → 121.8 | 5.1% | 138.4 → 128.5 | 7.7% |
| wide | 720×404 | 1087.0 → 1074.6 | 1.2% | 450.3 → 438.7 | 2.7% | 246.8 → 247.6 | -0.3% | 176.8 → 176.1 | 0.4% | 167.4 → 163.0 | 2.7% | 179.9 → 165.5 | 8.7% |
| park | 480×270 | 1910.7 → 1882.8 | 1.5% | 686.7 → 678.0 | 1.3% | 359.8 → 353.7 | 1.7% | 219.2 → 216.4 | 1.3% | 198.8 → 194.2 | 2.4% | 211.4 → 200.9 | 5.2% |
| town | 480×270 | 911.9 → 886.1 | 2.9% | 350.0 → 341.0 | 2.6% | 196.7 → 188.9 | 4.1% | 142.9 → 126.3 | 13.2% | 125.8 → 120.9 | 4.1% | 139.4 → 127.5 | 9.4% |
| caption | 480×270 | 550.8 → 519.5 | 6.0% | 227.6 → 221.7 | 2.6% | 135.6 → 128.3 | 5.7% | 100.2 → 92.8 | 8.0% | 94.9 → 88.7 | 7.0% | 106.3 → 97.8 | 8.7% |
| old2 | 480×360 | 1097.6 → 1072.1 | 2.4% | 433.3 → 421.8 | 2.7% | 249.3 → 235.7 | 5.8% | 170.7 → 159.1 | 7.3% | 153.1 → 139.7 | 9.6% | 160.9 → 147.8 | 8.8% |
| gradient | 480×270 | 188.1 → 180.1 | 4.5% | 93.9 → 91.6 | 2.5% | 63.9 → 61.2 | 4.5% | 57.8 → 57.0 | 1.4% | 55.5 → 55.6 | -0.1% | 63.1 → 60.3 | 4.6% |
| motion | 480×270 | 414.1 → 398.0 | 4.0% | 177.5 → 172.0 | 3.2% | 106.7 → 100.8 | 5.9% | 77.0 → 72.4 | 6.3% | 69.3 → 61.9 | 11.9% | 76.0 → 67.4 | 12.7% |
| short | 480×360 | 445.9 → 425.2 | 4.9% | 185.3 → 179.6 | 3.1% | 112.7 → 109.1 | 3.3% | 83.8 → 78.5 | 6.8% | 77.4 → 75.0 | 3.3% | 88.9 → 80.2 | 10.8% |
| long | 480×270 | 800.9 → 756.5 | 5.9% | 297.7 → 305.3 | -2.5% | 187.2 → 175.4 | 6.7% | 128.5 → 120.8 | 6.4% | 121.8 → 112.0 | 8.8% | 132.0 → 119.8 | 10.2% |

## Where the time went

Hardware counters from a separate set of runs on `old`, `wide`, `park` and `long`, five per binary per worker count, under `perf stat` and `/usr/bin/time` (whose own overhead inflates the wall times, so only the ratios below are meaningful). CPU is aggregate user + system seconds; RSS is process peak.

| Workers | Metric | Rayon | Pool | Change |
|---|---|---:|---:|---:|
| 1 | CPU seconds | 1.22 | 1.17 | -3.7% |
| 1 | task-clock (ms) | 1192 | 1152 | -3.4% |
| 1 | user cycles | 3.75e+09 | 3.67e+09 | -2.0% |
| 1 | user instructions | 8.58e+09 | 8.48e+09 | -1.2% |
| 1 | context switches | 376 | 326 | -13.4% |
| 1 | peak RSS (MiB) | 71.2 | 72.0 | +1.1% |
| 22 | CPU seconds | 2.29 | 2.17 | -5.5% |
| 22 | task-clock (ms) | 2263 | 2141 | -5.4% |
| 22 | user cycles | 5.05e+09 | 4.95e+09 | -2.1% |
| 22 | user instructions | 9.58e+09 | 9.44e+09 | -1.5% |
| 22 | context switches | 7341 | 1260 | -82.8% |
| 22 | peak RSS (MiB) | 98.2 | 99.9 | +1.7% |
| 44 | CPU seconds | 2.73 | 2.15 | -21.2% |
| 44 | task-clock (ms) | 2699 | 2118 | -21.5% |
| 44 | user cycles | 5.15e+09 | 4.78e+09 | -7.2% |
| 44 | user instructions | 9.83e+09 | 9.57e+09 | -2.7% |
| 44 | context switches | 33154 | 3908 | -88.2% |
| 44 | peak RSS (MiB) | 108.0 | 106.7 | -1.2% |

Instructions and user cycles barely move: the compute kernels are untouched. What falls is scheduling. Context switches drop by three quarters at 22 workers and by an order of magnitude at 44, and CPU seconds fall further than wall time does, so the same encode costs materially less machine, not just less time.

Peak RSS is unchanged. At one worker it differs by about a megabyte either way; at 22 and 44 the run-to-run spread within a single binary (for example 160-194 MiB across five Rayon runs of `wide` at 44) is several times the gap between the two.

## What replaced what

`src/pool.rs` is 415 lines of code excluding comments and tests. It exposes exactly the shapes weft uses — `for_each`, `map`, `map_init`, `for_each_mut`, and the two by-value forms `for_each_into` / `map_into` — plus a parallel entry sort for the one place `par_sort_unstable` was used. The differences that produce the numbers above:

- **One publish per region instead of a task graph.** A region stores one closure pointer and bumps an epoch; workers claim item ranges from a single atomic cursor. There are no deques, no task allocation, and no splitting.
- **Joining a region is optional.** A worker joins by incrementing the active count and re-reading the epoch, backing out if the region closed in between; the caller closes once it has drained the cursor itself and waits only for workers that joined. A descheduled or still-waking worker misses the region instead of stalling it. The first cut of this pool made every worker acknowledge every region, and a microbenchmark of empty regions showed why that cannot work at scale: at 16+ threads one late worker per region pushed every barrier past the caller's spin budget, the caller parked on a condvar, and a futex round trip on this VM is ~50 us — every region cost 52 us at 22 threads and 105 us at 44. With optional joining the same regions cost 2.5 us and 2.9 us.
- **Hot fields are on their own cache lines.** The epoch and job pointer that every worker polls never share a line with the active count that every worker RMWs, or with the per-worker parked flags.
- **The submitting thread is a worker.** `--threads N` is N runnable threads rather than N plus a blocked caller, and `--threads 1` spawns nothing and never synchronizes.
- **Worker indices are dense and always valid.** The quantize and LZW paths keep one scratch bundle per worker; under Rayon that needed `current_thread_index()` (a thread-local probe) plus a spare slot for the "ran on the calling thread" case. Now it is an array index the closure is handed.
- **`map_init` state is built once per worker, not once per split.** Rayon rebuilds it per split, which for the histogram strip scan meant reallocating the row and key buffers many times per batch.
- **Waiters spin, then poll on `yield_now`, then park.** The bounds are in microseconds rather than `pause` iterations, since a `pause` ranges from ~10 cycles on Broadwell to ~140 on Skylake and later. The yield phase is what keeps the pool out of the park/wake regime: a futex wake is tens of microseconds here, longer than most serial gaps between regions, so a worker that parked at 10 us would sleep through exactly the gaps it should cover. Yielding is free on an idle core and hands the core to the reader/hold threads on a busy one. A region unparks only as many workers as it has items for.
- **Adaptive spinning** (borrowed from filament, alex/claude-experiments#3): a worker whose previous idle episode ended in a park skips the spin/yield burn next time and parks at once. The case this is for is production's `ffmpeg | weft`, where the decoder is the bottleneck and the pool idles between batches. With `old` fed at 25 fps through a pipe (4.1 s wall for everyone), the yield burn showed up as 0.6 s of system time against Rayon's 0.4 s; with adaptive spinning the run costs 2.35 s of CPU (user + system) against Rayon's 2.7 s and 3.1 s without it, with involuntary context switches down from ~2,000 to ~50. On the cached-file benchmark it is neutral within noise; a 5 ms cold threshold was also measured and gave back part of the idle saving without a fast-path gain, so the threshold is 1 ms. A cold worker warms back up when a park turns out shorter than 1 ms (regions are arriving faster than the wake path can catch them) or when it wakes into a region already closed. Filament's other idle mechanisms — timed park ladders as a missed-wake backstop, and the caller blocking immediately after a long operation — were not taken: our wake handshake cannot lose a wake, and a caller that blocks immediately puts a futex wake on every long region's critical path, while a yielding caller costs an idle core nothing.
- **The entry sort is an MSD radix pass plus per-bucket comparison sorts**, preceded by a scan that returns immediately when the entries already arrive sorted, which is the common exact-histogram case. The per-bucket sort is on the full tuple, so the result is exactly what `sort_unstable` produces — the canonical order the palette depends on.

## Validation

- `cargo fmt --check`, `cargo clippy --release --all-targets -- -D warnings`.
- `cargo test --release`: 79 unit tests, up from 63. The sixteen new ones cover the pool: ordering and one-run-per-item at every scheduling boundary (0, 1, threads±1, 32·threads±1), per-worker init, by-value moves with a drop-counting type (each item and result dropped exactly once), nested regions, panics from items and from `init` on both the caller and worker paths, the entry sort against `sort_unstable`, parked workers waking and joining, 1,500 publishes jittered across the spin/yield/park windows (a lost wake-up hangs it), 50,000 single-item regions (late joiners backing out), four threads submitting to one pool concurrently, two pools driven concurrently, and drop with workers spinning, parked, or never started. Each runs across pools of 1, 2, 3, 8 and 33 workers where it makes sense. Integration tests: 13, up from 12.
- The end-to-end thread-invariance test now also runs 2 and 3 workers, and a new one drives a clip with more distinct colours than `FOLD_MAX` so the coarse-binned histogram and the parallel radix sort are both covered at 1, 2, 3, 5, 8 and 22 workers.
- The 896-run output-identity sweep described above.
