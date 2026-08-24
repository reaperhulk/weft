#!/usr/bin/env bash
# Fetch real-world benchmark clips (Simpsons frames) from frinkiac.com and
# decode them to y4m. Cartoon content is the workload weft is actually used
# on: large flat regions, hard edges, limited palettes, and — on older
# episodes — film grain. Synthetic clips (gen_inputs.sh) miss all of that.
#
# Renders each scene at 1/3/5/7/10 s so per-frame costs can be separated
# from per-clip fixed costs, plus 720p/1080p upscales of one scene so
# large-frame (cache / memory bandwidth) behavior is covered.
#
# Usage: bench/gen_frinkiac.sh [outdir]   (needs curl + ffmpeg)
set -euo pipefail
cd "$(dirname "$0")"
out=${1:-data/frink}
mkdir -p "$out"

# scene name -> "episode start_ms"; picked across eras (S07 is 1996 cel with
# film grain, S34 is modern flat digital) for content diversity.
scenes=(
  "hams S07E21 592217"
  "goo S16E12 677927"
  "mayor S29E06 131965"
  "cater S34E20 614739"
  "cupid S10E14 216425"
)
durs=(1 3 5 7 10)

render() { # episode start_ms end_ms -> prints path on frinkiac
  # The render endpoint transcodes on demand and occasionally returns an
  # error body instead of a url; back off and retry rather than aborting a
  # 25-clip fetch on one hiccup.
  local delay=2 body
  for _ in 1 2 3 4 5; do
    body=$(curl -sS -m 300 -X POST "https://frinkiac.com/api/render/mp4" \
      -H "Content-Type: application/json" \
      -d "[{\"episode\":\"$1\",\"start\":$2,\"end\":$3,\"overlays\":[]}]" || true)
    if url=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["url"])' 2>/dev/null); then
      printf '%s' "$url"; return 0
    fi
    sleep "$delay"; delay=$((delay * 2))
  done
  echo "render failed for $1 $2-$3: $body" >&2
  return 1
}

fetch() { # name episode start_ms dur_s
  local name=$1 ep=$2 start=$3 dur=$4
  local mp4="$out/$name-${dur}s.mp4"
  [ -s "$mp4" ] && return 0
  local url
  url=$(render "$ep" "$start" $((start + dur * 1000)))
  curl -sS -m 300 -o "$mp4" "https://frinkiac.com$url"
  sleep 1
}

for s in "${scenes[@]}"; do
  read -r name ep start <<<"$s"
  for d in "${durs[@]}"; do
    echo "fetching $name-${d}s..." >&2
    fetch "$name" "$ep" "$start" "$d"
    [ -s "$out/$name-${d}s.y4m" ] || \
      ffmpeg -v error -y -i "$out/$name-${d}s.mp4" -pix_fmt yuv420p "$out/$name-${d}s.y4m"
  done
done

# Upscaled variants: same content, big frames. Lanczos keeps real detail
# (a nearest-neighbour upscale would collapse the color count).
for spec in "720p:1280:720" "1080p:1920:1080"; do
  IFS=: read -r tag w h <<<"$spec"
  src="$out/hams-10s.mp4"
  dst="$out/hams-10s-$tag.y4m"
  [ -s "$dst" ] || ffmpeg -v error -y -i "$src" -vf "scale=$w:$h:flags=lanczos" \
    -pix_fmt yuv420p "$dst"
done

ls -la "$out"/*.y4m
