#!/usr/bin/env bash
# End-to-end raw RGBA -> GIF latency and peak-RSS benchmark using the fixed
# 1/3/5/7/10-second Frinkiac corpus and weft's defaults (including bluenoise).
set -euo pipefail
cd "$(dirname "$0")"

WEFT=${WEFT:-../target/release/weft}
RUNS=${RUNS:-5}
mkdir -p out/frinkiac
[ -s data/frinkiac/10s.rgba ] || ./fetch_frinkiac.sh
if [ ! -x "$WEFT" ]; then
  (cd .. && cargo build --release --quiet)
fi

printf "%-9s %10s %10s %11s\n" duration "best(ms)" "peak(MB)" "GIF(KB)"
for seconds in 1 3 5 7 10; do
  input="data/frinkiac/${seconds}s.rgba"
  output="out/frinkiac/${seconds}s.gif"
  best_ns= rss_kb=
  for run in $(seq "$RUNS"); do
    timing="${TMPDIR:-/tmp}/weft-frinkiac-$$-$run.time"
    start=$(date +%s%N)
    /usr/bin/time -f %M -o "$timing" "$WEFT" --format rgba \
      --size 640x360 --fps 30 < "$input" > "$output"
    elapsed=$(( $(date +%s%N) - start ))
    rss=$(cat "$timing")
    rm -f "$timing"
    if [ -z "$best_ns" ] || [ "$elapsed" -lt "$best_ns" ]; then
      best_ns=$elapsed
      rss_kb=$rss
    fi
  done
  printf "%-9s %10.2f %10.1f %11.1f\n" "${seconds}s" \
    "$(awk "BEGIN { print $best_ns / 1000000 }")" \
    "$(awk "BEGIN { print $rss_kb / 1024 }")" \
    "$(awk "BEGIN { print $(stat -c %s "$output") / 1024 }")"
done
