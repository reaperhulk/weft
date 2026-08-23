#!/usr/bin/env bash
# Benchmark weft against ffmpeg's palettegen/paletteuse pipeline.
# For each input clip, measures wall time, peak RSS, output size, and
# quality (PSNR/SSIM of the decoded GIF vs the source, both sampled at the
# source frame rate so duplicate-frame merging doesn't skew alignment).
#
# Usage: bench/run.sh [clip ...]   (default: all clips in bench/data)
set -euo pipefail
cd "$(dirname "$0")"

WEFT=${WEFT:-../target/release/weft}
RUNS=${RUNS:-3}
mkdir -p out

if [ ! -x "$WEFT" ]; then
  echo "building weft (release)..." >&2
  (cd .. && cargo build --release --quiet)
fi

[ -f data/testsrc.y4m ] || ./gen_inputs.sh

clips=("$@")
if [ ${#clips[@]} -eq 0 ]; then
  clips=()
  for f in data/*.y4m; do
    clips+=("$(basename "$f" .y4m)")
  done
fi

# time_cmd <out_time_var> <out_rss_var> -- cmd...
# Runs cmd $RUNS times, keeps best wall time (seconds) and its max RSS (KB).
time_cmd() {
  local best_t="" best_rss="" t rss
  for _ in $(seq "$RUNS"); do
    /usr/bin/time -f "%e %M" -o /tmp/weft_time.$$ "$@" || return 1
    read -r t rss < /tmp/weft_time.$$
    if [ -z "$best_t" ] || awk "BEGIN{exit !($t < $best_t)}"; then
      best_t=$t; best_rss=$rss
    fi
  done
  rm -f /tmp/weft_time.$$
  echo "$best_t $best_rss"
}

fps_of() {
  ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$1"
}

# quality <gif> <src.y4m> <fps> -> "psnr ssim"
# The GIF is resampled to the source rate so duplicate-frame merging doesn't
# skew alignment; both sides get sequential PTS before comparison.
quality() {
  local gif=$1 src=$2 fps=$3 psnr ssim
  psnr=$(ffmpeg -v info -i "$gif" -i "$src" -filter_complex \
    "[0:v]fps=${fps},format=rgb24,setpts=N[a];[1:v]format=rgb24,setpts=N[b];[a][b]psnr" \
    -f null - 2>&1 | grep -o 'average:[0-9.inf]*' | cut -d: -f2 || true)
  ssim=$(ffmpeg -v info -i "$gif" -i "$src" -filter_complex \
    "[0:v]fps=${fps},format=rgb24,setpts=N[a];[1:v]format=rgb24,setpts=N[b];[a][b]ssim" \
    -f null - 2>&1 | grep -o 'All:[0-9.]*' | cut -d: -f2 || true)
  echo "${psnr:-err} ${ssim:-err}"
}

printf "%-10s %-8s %8s %10s %10s %8s %8s\n" clip encoder "time(s)" "rss(MB)" "size(KB)" psnr ssim
printf '%.0s-' {1..68}; echo

# Apples-to-apples RGBA comparison: identical input bytes for both encoders,
# no YUV->RGB converter anywhere. The y4m rows below carry a caveat: the
# PSNR/SSIM reference is decoded with swscale, which truncates where weft
# rounds-to-nearest, so weft's flat regions show a constant +-1-2 offset
# against that reference (weft is the closer-to-exact conversion). This row
# is the honest quality comparison.
rgba_bench() {
  local raw=data/testsrc.rgba
  [ -f "$raw" ] || return 0
  local vs="-f rawvideo -pix_fmt rgba -video_size 640x360 -framerate 30"
  read -r ft frss < <(time_cmd ffmpeg -v error -y $vs -i "$raw" -filter_complex \
    "[0:v]split[a][b];[a]palettegen[p];[b][p]paletteuse" out/ffmpeg_rgba.gif)
  read -r wt wrss < <(time_cmd bash -c "$WEFT --size 640x360 --fps 30 < $raw > out/weft_rgba.gif")
  for enc in ffmpeg weft; do
    local gif=out/${enc}_rgba.gif t rss
    if [ $enc = ffmpeg ]; then t=$ft; rss=$frss; else t=$wt; rss=$wrss; fi
    local psnr ssim
    psnr=$(ffmpeg -v info -i "$gif" $vs -i "$raw" -filter_complex \
      "[0:v]fps=30,format=rgb24,setpts=N[a];[1:v]format=rgb24,setpts=N[b];[a][b]psnr" \
      -f null - 2>&1 | grep -o 'average:[0-9.inf]*' | cut -d: -f2 || true)
    ssim=$(ffmpeg -v info -i "$gif" $vs -i "$raw" -filter_complex \
      "[0:v]fps=30,format=rgb24,setpts=N[a];[1:v]format=rgb24,setpts=N[b];[a][b]ssim" \
      -f null - 2>&1 | grep -o 'All:[0-9.]*' | cut -d: -f2 || true)
    printf "%-10s %-8s %8s %10s %10s %8s %8s\n" "rgba" "$enc" "$t" \
      "$(awk "BEGIN{printf \"%.1f\", $rss/1024}")" \
      "$(awk "BEGIN{printf \"%d\", $(stat -c%s "$gif")/1024}")" "${psnr:-err}" "${ssim:-err}"
  done
}
rgba_bench

for clip in "${clips[@]}"; do
  src="data/$clip.y4m"
  fps=$(fps_of "$src")

  # ffmpeg baseline: single-command palettegen/paletteuse (in-memory two-pass).
  fgif="out/ffmpeg_$clip.gif"
  read -r ft frss < <(time_cmd ffmpeg -v error -y -i "$src" -filter_complex \
    "[0:v]split[a][b];[a]palettegen[p];[b][p]paletteuse" "$fgif")
  read -r fpsnr fssim < <(quality "$fgif" "$src" "$fps")
  printf "%-10s %-8s %8s %10s %10s %8s %8s\n" "$clip" ffmpeg "$ft" \
    "$(awk "BEGIN{printf \"%.1f\", $frss/1024}")" \
    "$(awk "BEGIN{printf \"%d\", $(stat -c%s "$fgif")/1024}")" "$fpsnr" "$fssim"

  # weft
  wgif="out/weft_$clip.gif"
  read -r wt wrss < <(time_cmd bash -c "$WEFT < '$src' > '$wgif'")
  read -r wpsnr wssim < <(quality "$wgif" "$src" "$fps")
  printf "%-10s %-8s %8s %10s %10s %8s %8s\n" "$clip" weft "$wt" \
    "$(awk "BEGIN{printf \"%.1f\", $wrss/1024}")" \
    "$(awk "BEGIN{printf \"%d\", $(stat -c%s "$wgif")/1024}")" "$wpsnr" "$wssim"
done
