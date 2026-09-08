#!/usr/bin/env python3
"""Build weft with PGO trained on the odd-numbered Frinkiac RGBA clips.

Accepts predecoded RGBA or the bundled MP4 corpus (requires ffmpeg).
Requires rustup's llvm-tools-preview component. The even-numbered clips are
reserved for checking the final binary against an optional ordinary build.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
MODES = [[], ["--lossy", "30", "--hold", "12"]]
CHECK_MODES = [*MODES, ["--colors", "18", "--dither", "none"]]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_clip(data, decoded, clip):
    """Check pinned inputs; decode once, outside training and verification."""
    name = clip["name"]
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError(f"Invalid clip name: {name!r}")
    raw = data / (name + ".rgba")
    if raw.exists():
        if sha256(raw) != clip["rgba_sha256"]:
            raise ValueError(f"RGBA checksum mismatch: {name}")
    else:
        mp4 = data / (name + ".mp4")
        if sha256(mp4) != clip["mp4_sha256"]:
            raise ValueError(f"MP4 checksum mismatch: {name}")
        raw = decoded / (name + ".rgba")
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "1",
                        "-i", str(mp4), "-map", "0:v:0", "-filter_threads", "1",
                        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt",
                        "rgba", str(raw)], check=True, timeout=60)
    expected = clip["width"] * clip["height"] * 4 * clip["frames"]
    if raw.stat().st_size != expected or expected <= 0:
        raise ValueError(f"Unexpected RGBA length: {name}")
    return raw


def encode(binary, raw, clip, threads, mode, output, env=None):
    with raw.open("rb") as src:
        subprocess.run([str(binary), "--format", "rgba", "--size",
                        f"{clip['width']}x{clip['height']}", "--fps", clip["fps"],
                        "--threads", str(threads), *mode], stdin=src,
                       stdout=output, check=True, env=env, timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "bench/out/arm64/weft-pgo")
    parser.add_argument("--threads", type=int,
                        default=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                        else os.cpu_count() or 1)
    parser.add_argument("--target", help="runnable Cargo target (for example aarch64-unknown-linux-musl)")
    parser.add_argument("--corpus", type=Path, default=ROOT / "bench/data/frinkiac")
    parser.add_argument("--baseline", type=Path,
                        help="ordinary binary to check against on held-out clips; copied before building")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output(["rustc", "-vV"], text=True)
    host = next(line.split(": ", 1)[1] for line in version.splitlines() if line.startswith("host: "))
    sysroot = Path(subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip())
    profdata = sysroot / "lib/rustlib" / host / "bin/llvm-profdata"
    if not profdata.is_file():
        parser.error("install the matching profile tools with: rustup component add llvm-tools-preview")
    data = args.corpus.resolve()
    manifest_path = data / "manifest.json"
    all_clips = json.loads(manifest_path.read_text())
    if not all_clips or len({clip["name"] for clip in all_clips}) != len(all_clips):
        parser.error("corpus must contain unique, nonempty clips")
    clips = all_clips[::2]
    held_out = all_clips[1::2] if args.baseline else []
    if args.baseline and not held_out:
        parser.error("baseline verification requires held-out clips")
    cargo = ["cargo", "rustc", "--locked", "--release", "--bin", "weft"]
    if args.target:
        cargo += ["--target", args.target]
    # The pool dispatches many closure types through the same call site.
    # LLVM's default value-profile counter reserve can fill during training.
    llvm_flags = ["-C", "llvm-args=-vp-counters-per-site=32"]
    timings = {}
    started = time.perf_counter()

    def stage(name, since):
        timings[name] = time.perf_counter() - since
        print(f"PGO {name}: {timings[name]:.2f}s", flush=True)

    # Fresh paths prevent both stale counters and Cargo reusing a binary
    # built against an older profile at the same profile-use path.
    with tempfile.TemporaryDirectory(prefix="pgo-", dir=output.parent) as tmp:
        profile = Path(tmp)
        baseline = profile / "baseline"
        if args.baseline:
            shutil.copy2(args.baseline.resolve(), baseline)
        raw_paths = {c["name"]: prepare_clip(data, profile, c) for c in [*clips, *held_out]}
        # FFmpeg versions can round color conversion differently. Pin the MP4
        # bytes and record the actual decoded bytes; both binaries see the same
        # RGBA. Predecoded inputs, above, must match their recorded checksum.
        raw_hashes = {name: sha256(raw) for name, raw in raw_paths.items()}
        stage("prepare", started)
        before = time.perf_counter()
        subprocess.run([*cargo, "--", *llvm_flags, "-C",
                        f"profile-generate={profile}"], cwd=ROOT, check=True)
        stage("instrument", before)
        target_dir = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
        if not target_dir.is_absolute():
            target_dir = ROOT / target_dir
        binary = target_dir / args.target / "release/weft" if args.target else target_dir / "release/weft"
        env = os.environ.copy()
        env["LLVM_PROFILE_FILE"] = str(profile / "%m-%p.profraw")
        before = time.perf_counter()
        for mode in MODES:
            for clip in clips:
                encode(binary, raw_paths[clip["name"]], clip, args.threads, mode,
                       subprocess.DEVNULL, env)
            print(f"Trained {len(clips)} clips: {mode or 'defaults'}", flush=True)
        stage("train", before)
        before = time.perf_counter()
        merged = profile / "weft.profdata"
        profiles = sorted(profile.glob("*.profraw"))
        if not profiles or any(p.stat().st_size == 0 for p in profiles):
            raise RuntimeError("PGO training produced missing or empty profiles")
        subprocess.run([str(profdata), "merge", "-o", str(merged),
                        *map(str, profiles)], check=True)
        stage("merge", before)
        before = time.perf_counter()
        subprocess.run([*cargo, "--", *llvm_flags, "-C",
                        f"profile-use={merged}", "-C", "llvm-args=-pgo-warn-missing-function"],
                       cwd=ROOT, check=True)
        stage("optimize", before)
        before = time.perf_counter()
        for clip in held_out:
            for mode in CHECK_MODES:
                hashes = []
                for candidate in (baseline, binary):
                    gif = profile / "check.gif"
                    with gif.open("wb") as dst:
                        encode(candidate, raw_paths[clip["name"]], clip, args.threads, mode, dst)
                    hashes.append(sha256(gif))
                if hashes[0] != hashes[1]:
                    raise RuntimeError(f"PGO output mismatch: {clip['name']}, {mode}")
        stage("verify", before)
        if held_out:
            print(f"Verified {len(held_out) * len(CHECK_MODES)} held-out outputs", flush=True)
        if output != binary:
            shutil.copy2(binary, output)
        shutil.copy2(merged, output.with_suffix(".profdata"))
    stage("total", started)
    metadata = dict(
        rustc=version, target=args.target or host, threads=args.threads,
        modes=MODES, training_clips=clips, llvm_flags=llvm_flags,
        corpus_manifest_sha256=sha256(manifest_path), rgba_sha256=raw_hashes,
        verified_clips=held_out, check_modes=CHECK_MODES if held_out else [],
        timings_seconds=timings, binary_sha256=sha256(output),
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as dst:
            dst.write(f"### PGO: {args.target or host}\n\n| Stage | Seconds |\n|---|---:|\n")
            for name, seconds in timings.items():
                dst.write(f"| {name} | {seconds:.2f} |\n")
            dst.write(f"\nVerified {len(held_out) * len(CHECK_MODES)} held-out outputs.\n")
    print(f"PGO binary: {output}")


if __name__ == "__main__":
    main()
