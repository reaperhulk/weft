#!/usr/bin/env python3
"""Fetch 20 reproducible Frinkiac MP4 excerpts and decode once to raw RGBA.

Requires curl, ffmpeg and ffprobe. Media and generated metadata are gitignored.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "frinkiac"
# Spread across old cel animation and newer digital/HD episodes, with varied
# scene positions and durations. No captions, resizing or frame-rate conversion.
CLIPS = [
    ("S01E01", 300000, 2000), ("S02E04", 450000, 2500),
    ("S03E08", 600000, 3000), ("S04E12", 750000, 3500),
    ("S05E02", 900000, 4000), ("S06E06", 350000, 4500),
    ("S07E10", 500000, 5000), ("S08E14", 650000, 5500),
    ("S09E03", 800000, 6000), ("S10E07", 950000, 6500),
    ("S11E11", 400000, 7000), ("S12E15", 550000, 7500),
    ("S14E02", 700000, 8000), ("S16E06", 850000, 8500),
    ("S18E10", 1000000, 9000), ("S20E14", 425000, 9500),
    ("S23E03", 575000, 9900), ("S27E07", 725000, 2200),
    ("S31E11", 875000, 6200), ("S35E15", 1025000, 9800),
]


def capture(*args):
    return subprocess.check_output(args)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (episode, start, duration) in enumerate(CLIPS, 1):
        name = f"{i:02d}-{episode}-{duration}ms"
        mp4 = DATA / f"{name}.mp4"
        raw = DATA / f"{name}.rgba"
        request = dict(episode=episode, start=start, end=start + duration, overlays=[])
        receipt = DATA / f"{name}.jsonl"
        if not mp4.exists():
            # Render requests are expensive; stay well below five per minute.
            # Retry throttling/server failures with a full minute of backoff.
            time.sleep(20)
            response = capture("curl", "-sSfL", "--retry", "3", "--retry-all-errors",
                               "--retry-delay", "60", "--max-time", "180",
                               "https://frinkiac.com/api/render/mp4/stream",
                               "-H", "Content-Type: application/json",
                               "--data", json.dumps([request]))
            receipt.write_bytes(response)
            urls = [r["url"] for line in response.splitlines()
                    if "url" in (r := json.loads(line))]
            if not urls:
                raise RuntimeError(f"No video URL for {name}: {response!r}")
            url = "https://frinkiac.com" + urls[-1]
            partial = mp4.with_suffix(".part")
            subprocess.run(["curl", "-sSfL", "--retry", "3", "--max-time", "180",
                            url, "-o", str(partial)], check=True)
            partial.replace(mp4)
        stream = json.loads(capture("ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                    "-show_streams", "-of", "json", str(mp4)))["streams"][0]
        if not raw.exists():
            partial = raw.with_suffix(".part")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4),
                            "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo",
                            "-pix_fmt", "rgba", str(partial)], check=True)
            partial.replace(raw)
        frame_bytes = stream["width"] * stream["height"] * 4
        assert raw.stat().st_size % frame_bytes == 0
        frames = raw.stat().st_size // frame_bytes
        assert frames == int(stream["nb_read_frames"])
        entry = dict(name=name, request=request, width=stream["width"], height=stream["height"],
                     fps=stream["r_frame_rate"], duration=float(stream["duration"]), frames=frames,
                     mp4_sha256=hashlib.sha256(mp4.read_bytes()).hexdigest(),
                     rgba_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
        manifest.append(entry)
        (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{name}: {entry['width']}x{entry['height']}, {frames} frames, {entry['duration']:.3f}s", flush=True)
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
