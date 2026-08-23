#!/usr/bin/env bash
# Generate benchmark/test inputs with ffmpeg lavfi sources.
# Outputs land in bench/data/ (gitignored). Idempotent: skips existing files.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data

gen() {
  local name=$1
  shift
  if [ ! -f "data/$name.y4m" ]; then
    echo "generating $name.y4m"
    ffmpeg -v error -y "$@" -pix_fmt yuv420p "data/$name.y4m"
  fi
}

# Synthetic motion + text + shapes: the classic encoder workout.
gen testsrc   -f lavfi -i "testsrc2=size=640x360:rate=30:duration=5"
# Smooth gradients everywhere: worst case for palette quality / banding.
gen mandel    -f lavfi -i "mandelbrot=size=640x360:rate=30" -t 5
# Animated soft gradients: dithering quality test.
gen gradients -f lavfi -i "gradients=size=640x360:rate=30:speed=0.2:duration=5"
# Mostly-static frames: tests inter-frame delta + duplicate-frame merging.
gen static    -f lavfi -i "smptehdbars=size=640x360:rate=30:duration=5"
# High-entropy organic motion.
gen life      -f lavfi -i "life=size=320x180:rate=30:mold=10:ratio=0.1:death_color=#c83232:life_color=#00ff00" -vf scale=640:360:flags=neighbor -t 5
# Scale test: 720p, 10 seconds.
gen big       -f lavfi -i "testsrc2=size=1280x720:rate=30:duration=10"

# Raw RGBA variant of one clip, for exercising the rgba input path.
if [ ! -f data/testsrc.rgba ]; then
  echo "generating testsrc.rgba (640x360, 30fps)"
  ffmpeg -v error -y -i data/testsrc.y4m -f rawvideo -pix_fmt rgba data/testsrc.rgba
fi

echo "done. inputs in $(pwd)/data"
