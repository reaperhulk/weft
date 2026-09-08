# Release PGO

CI builds each of the four Linux release targets with a fresh PGO profile on
its native architecture. The jobs run in parallel. The existing Cargo cache
reuses dependencies; only the weft binary crate is instrumented and rebuilt.
No profile from another commit, toolchain or target is reused.

The entire corpus is bundled here: **6,060,848 bytes of MP4 video**, plus the
manifest. CI makes no Frinkiac requests. FFmpeg decodes the clips to temporary
RGBA files once per job; neither FFmpeg nor the corpus is needed by the shipped
binary. There is no large decoded-data cache to transfer or external corpus
service to provision.

The builder trains on the ten odd-numbered clips, once with defaults and once
with `--lossy 30 --hold 12`, using the runner's available CPU count. It then
compares the final binary against the ordinary executable produced by
`cargo test`, on the ten even-numbered clips in three modes: defaults,
lossy/hold, and `--colors 18 --dither none`. All 30 GIFs must match byte for byte.
The held-out runs do not contribute to the profile.

Each build step has a five-minute timeout; the complete binary job has an
eight-minute timeout. Exceeding either fails the job and prevents the release
upload. These are failure limits, not promised build durations. Per-stage
timings appear in the Actions summary. Profiles and JSON metadata are retained
as separate Actions artifacts for seven days; release assets remain just the
binaries, with the existing static/glibc compatibility checks.

## Local validation

On the M1 Max with Rust 1.91.1 and two encoding workers, a cold Cargo target
directory took **62.0 seconds** for the ordinary build, 95 Rust tests, corpus
preparation, PGO build, and held-out verification:

| Stage | Seconds |
|---|---:|
| Cold ordinary build and tests | 14.86 |
| Decode and verify corpus | 3.34 |
| Instrumented build | 8.60 |
| Train: 20 encodes | 9.53 |
| Merge profiles | 0.01 |
| Optimized build | 7.75 |
| Verify: 30 pairs of encodes | 17.82 |

This excludes checkout, installing Rust/LLVM tools/FFmpeg, cache transfer and
artifact upload. It is a local timing, not a measurement of the Linux runners
or CI's pinned Rust 1.98 toolchain. The first Actions run establishes those
numbers; performance gains must also be evaluated per Linux target.

A separate paired, shuffled benchmark (20 clips, five runs, ten encoding
workers) measured 81.84 ms for the ordinary build and 79.16 ms for this
two-worker-trained PGO build: **3.4% higher throughput**, with identical GIFs.
The earlier ten-worker-trained PGO build measured 77.98 ms in the same run.
Training worker count can affect the resulting profile; the workflow uses
available CPUs rather than assuming this Mac's ten cores.

To exercise the same pipeline locally (FFmpeg must be installed):

```sh
rustup component add llvm-tools-preview
cargo test --locked --release
python3 bench/build_pgo.py --corpus bench/pgo/corpus \
  --baseline target/release/weft --output bench/out/pgo/weft
```

CI additionally passes `--target` and uses the corresponding target-specific
executable paths. Both build phases preserve the repository's existing CPU
settings: x86-64-v3 for x86_64, baseline ARM64 for aarch64. The profile process
follows the [Rust PGO workflow](https://doc.rust-lang.org/rustc/profile-guided-optimization.html).

## Corpus provenance and updates

The MP4s are the unmodified 20 Frinkiac excerpts used in the
[ARM64 hill climb](../arm64-hill-climb.md), fetched on 2026-09-08. They contain
third-party Simpsons footage; the repository's Apache-2.0 software license
does not grant rights to that footage. `corpus/manifest.json` records episode
windows, dimensions, frame counts and SHA-256 hashes. The builder verifies MP4
hashes before decoding and checks decoded byte counts. FFmpeg versions may
round color conversion differently, so it records actual RGBA hashes and
compares both executables on the same decoded input. Predecoded RGBA inputs
must match the manifest's RGBA checksum.

To refresh the bundle deliberately, use `bench/fetch_frinkiac.py` (which paces
render requests and backs off on throttling), copy its 20 MP4s and manifest
from `bench/data/frinkiac/` into `bench/pgo/corpus/`, and repeat held-out
correctness and performance measurements. Do not fetch or change the corpus
as part of a release build.
