#!/usr/bin/env bash
# Fetch the fixed real-world RGBA benchmark corpus. The URLs below are
# Frinkiac's permanent generated-video URLs: this script deliberately never
# calls the generation API, so repeated setup does not POST or create videos.
set -euo pipefail
cd "$(dirname "$0")"

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }
mkdir -p data/frinkiac

# One continuous cartoon scene, cut server-side to each benchmark duration.
# Keeping the start timestamp fixed makes duration scaling comparable.
episode=S05E15
start=292324
for seconds in 1 3 5 7 10; do
  end=$((start + seconds * 1000))
  mp4="data/frinkiac/${seconds}s.mp4"
  rgba="data/frinkiac/${seconds}s.rgba"
  url="https://frinkiac.com/video/$episode/$start/$end.mp4"
  if [ ! -s "$mp4" ]; then
    echo "fetching $url" >&2
    curl --fail --location --retry 3 --output "$mp4.tmp" "$url"
    mv "$mp4.tmp" "$mp4"
  fi
  if [ ! -s "$rgba" ]; then
    echo "decoding ${seconds}s RGBA fixture" >&2
    ffmpeg -v error -y -i "$mp4" -vf "scale=640:360:flags=lanczos,fps=30" \
      -frames:v $((seconds * 30)) -f rawvideo -pix_fmt rgba "$rgba.tmp"
    expected=$((seconds * 30 * 640 * 360 * 4))
    actual=$(stat -c %s "$rgba.tmp")
    if [ "$actual" -ne "$expected" ]; then
      echo "unexpected decoded size: got $actual, expected $expected" >&2
      rm -f "$rgba.tmp"
      exit 1
    fi
    mv "$rgba.tmp" "$rgba"
  fi
done

echo "Frinkiac fixtures ready in $(pwd)/data/frinkiac" >&2
