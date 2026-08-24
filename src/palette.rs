//! Global palette generation and exact nearest-color lookup, both operating
//! in OkLab space to match ffmpeg's palettegen/paletteuse (ffmpeg >= 5.x).
//!
//! - Histogram: open-addressed hash keyed by exact 24-bit color, so sources
//!   with few distinct colors get a lossless palette. When the deduped
//!   histogram outgrows the 6-bit/channel bin count (true-color content),
//!   it is folded to one count-weighted mean color per bin before median
//!   cut — a 255-color palette can't resolve finer than that, and median
//!   cut's cost is linear in distinct colors.
//! - Median cut: variance-based (Heckbert), palettegen-style — the box with
//!   the largest single-channel squared error (in Lab) is split at its
//!   count-weighted median along that channel; each box yields the Lab
//!   average of its colors, converted back to sRGB.
//! - Nearest lookup: per-cell candidate lists over a 6-bit/channel RGB grid
//!   (locally sorted search). Distances are OkLab; lists are built with a
//!   triangle-inequality bound so the argmin over candidates is the true
//!   nearest for every integer color in the cell. Most cells end up with a
//!   single candidate, collapsing lookup to one load.

use crate::oklab::{oklab_to_srgb, LabConverter};
use rayon::prelude::*;

pub const GRID_BITS: u32 = 6;
pub const GRID_SIZE: usize = 1 << (3 * GRID_BITS); // 262144

/// Bits per channel for the histogram fold. Separate from the lookup
/// grid: the fold array is written randomly once per colour run, so its
/// size is what pass 1 pays in cache misses, while the lookup grid is
/// read-mostly.
pub const FOLD_BITS: u32 = 5;
pub const FOLD_SIZE: usize = 1 << (3 * FOLD_BITS);

#[inline(always)]
pub fn fold_key(r: u8, g: u8, b: u8) -> usize {
    let sh = 8 - FOLD_BITS;
    (((r as usize) >> sh) << (2 * FOLD_BITS))
        | (((g as usize) >> sh) << FOLD_BITS)
        | ((b as usize) >> sh)
}

#[inline(always)]
pub fn grid_key(r: u8, g: u8, b: u8) -> usize {
    (((r as usize) >> 2) << (2 * GRID_BITS))
        | (((g as usize) >> 2) << GRID_BITS)
        | ((b as usize) >> 2)
}

// ---------------------------------------------------------------------------
// Exact-color histogram

const EMPTY: u32 = u32::MAX;

pub struct ColorHist {
    keys: Vec<u32>, // 24-bit color or EMPTY
    counts: Vec<u32>,
    len: usize,
    mask: usize,
}

impl ColorHist {
    pub fn new() -> Self {
        let cap = 1 << 16;
        ColorHist {
            keys: vec![EMPTY; cap],
            counts: vec![0; cap],
            len: 0,
            mask: cap - 1,
        }
    }

    #[inline(always)]
    fn slot_of(&self, color: u32) -> usize {
        let mut slot = (color.wrapping_mul(0x9E37_79B1) as usize >> 8) & self.mask;
        loop {
            let k = self.keys[slot];
            if k == color || k == EMPTY {
                return slot;
            }
            slot = (slot + 1) & self.mask;
        }
    }

    /// Prefetch the home slot for `color` (the add loop runs a few runs
    /// ahead of itself; the table is large enough that every probe is a
    /// cache miss without this).
    #[inline(always)]
    pub fn prefetch(&self, color: u32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let slot = (color.wrapping_mul(0x9E37_79B1) as usize >> 8) & self.mask;
            _mm_prefetch(self.keys.as_ptr().add(slot) as *const i8, _MM_HINT_T0);
            _mm_prefetch(self.counts.as_ptr().add(slot) as *const i8, _MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = color;
        }
    }

    #[inline(always)]
    pub fn add(&mut self, color: u32, n: u32) {
        let slot = self.slot_of(color);
        if self.keys[slot] == EMPTY {
            self.keys[slot] = color;
            self.counts[slot] = n;
            self.len += 1;
            if self.len * 10 > self.keys.len() * 7 {
                self.grow();
            }
        } else {
            self.counts[slot] = self.counts[slot].saturating_add(n);
        }
    }

    fn grow(&mut self) {
        let new_cap = self.keys.len() * 2;
        let old_keys = std::mem::replace(&mut self.keys, vec![EMPTY; new_cap]);
        let old_counts = std::mem::replace(&mut self.counts, vec![0; new_cap]);
        self.mask = new_cap - 1;
        for (k, c) in old_keys.into_iter().zip(old_counts) {
            if k != EMPTY {
                let slot = self.slot_of(k);
                self.keys[slot] = k;
                self.counts[slot] = c;
            }
        }
    }

    /// Number of distinct colors held.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn entries(&self) -> Vec<(u32, u32)> {
        self.keys
            .iter()
            .zip(self.counts.iter())
            .filter(|(&k, _)| k != EMPTY)
            .map(|(&k, &c)| (k, c))
            .collect()
    }
}

/// Accumulate one RGBA frame into the exact-color histogram. Pixels with
/// alpha < 128 are skipped; returns true when any were, so the pipeline
/// knows the disposal mode (and thus whether delta encoding is possible)
/// before quantization starts. `runs` is caller-provided scratch: the row
/// is RLE-scanned into it first, then added with the table slot for a few
/// runs ahead prefetched — the adds are otherwise serialized behind one
/// cache miss each on noisy content.
pub fn accumulate_frame(hist: &mut ColorHist, rgba: &[u8], runs: &mut Vec<(u32, u32)>) -> bool {
    let has_alpha = scan_runs(rgba, runs);
    for j in 0..runs.len() {
        if let Some(&(c, _)) = runs.get(j + 8) {
            hist.prefetch(c);
        }
        let (c, n) = runs[j];
        hist.add(c, n);
    }
    has_alpha
}

/// Accumulate one RGBA frame directly into 6-bit/channel bins (the coarse
/// mode workers switch to once their exact table outgrows the spill
/// threshold — see `maybe_fold` for why the grid is quality-sufficient).
/// A binned add is a key computation plus four u64 adds, with no probing
/// and no table growth, and bin sums are commutative, so the result is
/// independent of how frames are scheduled across workers. Same
/// prefetched two-pass shape as `accumulate_frame` (the bin array is 8MB).
pub fn accumulate_frame_coarse(
    bins: &mut [[u64; 4]],
    rgba: &[u8],
    runs: &mut Vec<(u32, u32)>,
) -> bool {
    let has_alpha = scan_runs(rgba, runs);
    for j in 0..runs.len() {
        if let Some(&(c, _)) = runs.get(j + 8) {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                let key = fold_key((c >> 16) as u8, (c >> 8) as u8, c as u8);
                _mm_prefetch(bins.as_ptr().add(key) as *const i8, _MM_HINT_T0);
            }
            #[cfg(not(target_arch = "x86_64"))]
            let _ = c;
        }
        let (c, n) = runs[j];
        bin_add(bins, c, n);
    }
    has_alpha
}

/// Shared RLE scan into `runs` (cleared first): run-length batching keeps
/// sink traffic low on flat content.
#[inline(always)]
fn scan_runs(rgba: &[u8], runs: &mut Vec<(u32, u32)>) -> bool {
    if let Some(alpha) = crate::simdops::scan_runs(crate::simdops::level(), rgba, runs) {
        return alpha;
    }
    runs.clear();
    scan_runs_with(rgba, |c, n| runs.push((c, n)))
}

#[inline(always)]
fn scan_runs_with(rgba: &[u8], mut add: impl FnMut(u32, u32)) -> bool {
    let pixels = rgba.as_chunks::<4>().0;
    let n = pixels.len();
    let mut has_alpha = false;
    let mut last: u32 = u32::MAX;
    let mut run: u32 = 0;
    let mut i = 0usize;
    while i < n {
        let px = pixels[i];
        if px[3] < 128 {
            has_alpha = true;
            i += 1;
            continue;
        }
        let c = ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | px[2] as u32;
        if c == last {
            run += 1;
        } else {
            if run > 0 {
                add(last, run);
            }
            last = c;
            run = 1;
        }
        i += 1;
        // Bulk-extend the run in 8-pixel blocks (32-byte compares lower
        // to SIMD memcmp). Guarded by one scalar pixel compare so noisy
        // content doesn't pay the pattern-building cost per pixel; equal
        // raw words share alpha, so every counted pixel is opaque.
        if i < n && pixels[i] == px {
            let mut pattern = [0u8; 32];
            for chunk in pattern.as_chunks_mut::<4>().0 {
                *chunk = px;
            }
            let bytes = &rgba[i * 4..];
            let mut off = 0usize;
            while off + 32 <= bytes.len() && bytes[off..off + 32] == pattern {
                off += 32;
            }
            run += (off / 4) as u32;
            i += off / 4;
        }
    }
    if run > 0 {
        add(last, run);
    }
    has_alpha
}

// ---------------------------------------------------------------------------
// Histogram folding

/// Fold an exact-color histogram into `GRID_BITS`-per-channel bins, each
/// represented by the count-weighted mean of its colors, when it holds more
/// entries than there are bins (so folding always shrinks). Means stay
/// within their (4-wide) cell, so output colors remain unique — a property
/// `median_cut`'s tie-breaking relies on.
pub fn maybe_fold(entries: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    if entries.len() <= FOLD_SIZE {
        return entries;
    }
    let mut bins = new_fold_bins();
    fold_into_bins(&mut bins, &entries);
    fold_bins_to_entries(&bins)
}

/// A zeroed bin array: `[count, r_sum, g_sum, b_sum]` per grid cell.
pub fn new_fold_bins() -> Vec<[u64; 4]> {
    vec![[0u64; 4]; FOLD_SIZE]
}

#[inline(always)]
fn bin_add(bins: &mut [[u64; 4]], c: u32, n: u32) {
    let (r, g, b) = (c >> 16, (c >> 8) & 255, c & 255);
    let bin = &mut bins[fold_key(r as u8, g as u8, b as u8)];
    let n = n as u64;
    bin[0] += n;
    bin[1] += n * r as u64;
    bin[2] += n * g as u64;
    bin[3] += n * b as u64;
}

/// Sum exact-color entries into the bin array.
pub fn fold_into_bins(bins: &mut [[u64; 4]], entries: &[(u32, u32)]) {
    for &(c, n) in entries {
        bin_add(bins, c, n);
    }
}

/// Merge `src`'s sums into `dst` (worker bin arrays at reduce time).
pub fn merge_bins(dst: &mut [[u64; 4]], src: &[[u64; 4]]) {
    for (d, s) in dst.iter_mut().zip(src) {
        for c in 0..4 {
            d[c] += s[c];
        }
    }
}

/// Emit one (mean color, count) entry per populated bin, in bin order.
pub fn fold_bins_to_entries(bins: &[[u64; 4]]) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for bin in bins {
        let n = bin[0];
        if n == 0 {
            continue;
        }
        let r = ((bin[1] + n / 2) / n) as u32;
        let g = ((bin[2] + n / 2) / n) as u32;
        let b = ((bin[3] + n / 2) / n) as u32;
        out.push(((r << 16) | (g << 8) | b, n.min(u32::MAX as u64) as u32));
    }
    out
}

// ---------------------------------------------------------------------------
// Median cut (in OkLab)

#[derive(Clone, Copy)]
struct HBin {
    count: u32,
    srgb: u32,
    lab: [f32; 3],
}

struct Box_ {
    start: usize,
    len: usize,
    count: u64,
    cut_score: f64,
    cut_axis: usize,
}

/// Boxes at least this long compute their sums with parallel fixed-size
/// chunks (only the first few splits of a large histogram qualify). The
/// per-chunk partials combine sequentially in chunk order, so the result
/// is independent of thread scheduling — it differs from the serial
/// left-fold only in rounding grouping, which is equally arbitrary.
const PAR_BOX: usize = 1 << 16;
const BOX_CHUNK: usize = 8192;

fn make_box(bins: &[HBin], start: usize, len: usize) -> Box_ {
    let slice = &bins[start..start + len];
    let (count, sum) = if len >= PAR_BOX {
        slice
            .par_chunks(BOX_CHUNK)
            .map(|ch| {
                let mut count = 0u64;
                let mut sum = [0f64; 3];
                for b in ch {
                    count += b.count as u64;
                    for (s, &l) in sum.iter_mut().zip(&b.lab) {
                        *s += b.count as f64 * l as f64;
                    }
                }
                (count, sum)
            })
            .collect::<Vec<_>>()
            .iter()
            .fold((0u64, [0f64; 3]), |(ac, asum), &(c, s)| {
                (ac + c, [asum[0] + s[0], asum[1] + s[1], asum[2] + s[2]])
            })
    } else {
        let mut count = 0u64;
        let mut sum = [0f64; 3];
        for b in slice {
            count += b.count as u64;
            for (s, &l) in sum.iter_mut().zip(&b.lab) {
                *s += b.count as f64 * l as f64;
            }
        }
        (count, sum)
    };
    let mean = [
        sum[0] / count as f64,
        sum[1] / count as f64,
        sum[2] / count as f64,
    ];
    let er2 = if len >= PAR_BOX {
        slice
            .par_chunks(BOX_CHUNK)
            .map(|ch| {
                let mut er2 = [0f64; 3];
                for b in ch {
                    for c in 0..3 {
                        let d = b.lab[c] as f64 - mean[c];
                        er2[c] += b.count as f64 * d * d;
                    }
                }
                er2
            })
            .collect::<Vec<_>>()
            .iter()
            .fold([0f64; 3], |a, e| [a[0] + e[0], a[1] + e[1], a[2] + e[2]])
    } else {
        let mut er2 = [0f64; 3];
        for b in slice {
            for c in 0..3 {
                let d = b.lab[c] as f64 - mean[c];
                er2[c] += b.count as f64 * d * d;
            }
        }
        er2
    };
    // palettegen: split axis and box choice both follow the single largest
    // per-channel squared error
    let mut cut_axis = 0;
    for c in 1..3 {
        if er2[c] > er2[cut_axis] {
            cut_axis = c;
        }
    }
    Box_ {
        start,
        len,
        count,
        cut_score: er2[cut_axis],
        cut_axis,
    }
}

/// Packed split key: sortable-f32 of lab[axis] in the high bits, srgb
/// tiebreak in the low 24. Unsigned order on the key equals
/// `lab[axis].partial_cmp(..).then(srgb.cmp(..))` — keys are unique
/// because srgb values are.
#[inline(always)]
fn axis_key(b: &HBin, axis: usize) -> u64 {
    // +0.0 folds -0.0 onto +0.0 so key order matches partial_cmp, which
    // treats the two zeros as equal (the srgb tiebreak decides there).
    let bits = (b.lab[axis] + 0.0).to_bits();
    let sortable = if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    };
    ((sortable as u64) << 24) | b.srgb as u64
}

/// Find the count-weighted median split of a box without sorting it:
/// returns (s, k) where s is the smallest prefix length (in key order)
/// whose count sum exceeds `median`, and k is the s-th smallest key.
/// Quickselect-style partitioning: expected O(len), vs the O(len log len)
/// comparator sort it replaces. Keys and counts travel as parallel arrays
/// so the AVX-512 path can compress-store eight elements per instruction;
/// elements below the pivot stream into the spare buffer while the rest
/// compact in place (the write cursor never passes the read cursor). The
/// result is value-defined — the smallest prefix in key order whose count
/// sum exceeds `median`, and the key at that boundary — so pivot choices
/// and intermediate elemenet order never affect the outcome.
///
/// One partition round: `< pivot` goes to (ko, co) from index 0, the rest
/// compacts in place at (kc, cc); returns the left count and its weight.
fn partition_scalar(
    kc: &mut [u64],
    cc: &mut [u32],
    ko: &mut [u64],
    co: &mut [u32],
    pivot: u64,
) -> (usize, u64) {
    let n = kc.len();
    let mut l = 0usize;
    let mut r = 0usize;
    let mut wl = 0u64;
    for i in 0..n {
        let k = kc[i];
        let c = cc[i];
        let less = k < pivot;
        // cmov-friendly: unconditional store on each side's cursor
        ko[l] = k;
        co[l] = c;
        kc[r] = k;
        cc[r] = c;
        l += less as usize;
        r += !less as usize;
        wl += c as u64 * less as u64;
    }
    (l, wl)
}

/// AVX-512 partition round: 8 keys per iteration via masked
/// compress-stores. Same contract as `partition_scalar`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl")]
unsafe fn partition_avx512(
    kc: &mut [u64],
    cc: &mut [u32],
    ko: &mut [u64],
    co: &mut [u32],
    pivot: u64,
) -> (usize, u64) {
    use std::arch::x86_64::*;
    let n = kc.len();
    let kcp = kc.as_mut_ptr();
    let ccp = cc.as_mut_ptr();
    let kop = ko.as_mut_ptr();
    let cop = co.as_mut_ptr();
    let pv = _mm512_set1_epi64(pivot as i64);
    let mut l = 0usize;
    let mut r = 0usize;
    let mut wacc = _mm512_setzero_si512();
    let mut i = 0usize;
    while i + 8 <= n {
        let kv = _mm512_loadu_si512(kcp.add(i) as *const _);
        let cv = _mm256_loadu_si256(ccp.add(i) as *const _);
        let m = _mm512_cmplt_epu64_mask(kv, pv);
        _mm512_mask_compressstoreu_epi64(kop.add(l) as *mut _, m, kv);
        _mm256_mask_compressstoreu_epi32(cop.add(l) as *mut _, m, cv);
        _mm512_mask_compressstoreu_epi64(kcp.add(r) as *mut _, !m, kv);
        _mm256_mask_compressstoreu_epi32(ccp.add(r) as *mut _, !m, cv);
        wacc = _mm512_add_epi64(wacc, _mm512_maskz_cvtepu32_epi64(m, cv));
        let lc = m.count_ones() as usize;
        l += lc;
        r += 8 - lc;
        i += 8;
    }
    let mut wl = _mm512_reduce_add_epi64(wacc) as u64;
    while i < n {
        let k = *kcp.add(i);
        let c = *ccp.add(i);
        if k < pivot {
            *kop.add(l) = k;
            *cop.add(l) = c;
            l += 1;
            wl += c as u64;
        } else {
            *kcp.add(r) = k;
            *ccp.add(r) = c;
            r += 1;
        }
        i += 1;
    }
    (l, wl)
}

#[cfg(target_arch = "x86_64")]
fn has_avx512() -> bool {
    use std::sync::OnceLock;
    static B: OnceLock<bool> = OnceLock::new();
    *B.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl")
    })
}

fn weighted_split(
    keys: &mut [u64],
    counts: &mut [u32],
    tkeys: &mut [u64],
    tcounts: &mut [u32],
    median: u64,
) -> (usize, u64) {
    let mut in_cur = true; // current range lives in `keys` (else `tkeys`)
    let mut n = keys.len();
    let mut base = 0usize; // elements finalized to the left of the range
    let mut acc = 0u64; // their count sum
    loop {
        // Invariant: acc <= median and the boundary lies inside the range,
        // which always starts at index 0 of its buffer (both partition
        // outputs are written from the front).
        let (kc, ko, cc, co) = if in_cur {
            (&mut *keys, &mut *tkeys, &mut *counts, &mut *tcounts)
        } else {
            (&mut *tkeys, &mut *keys, &mut *tcounts, &mut *counts)
        };
        if n <= 64 {
            let mut sub = [(0u64, 0u32); 64];
            for (s, (&k, &c)) in sub.iter_mut().zip(kc[..n].iter().zip(&cc[..n])) {
                *s = (k, c);
            }
            let sub = &mut sub[..n];
            sub.sort_unstable();
            for (i, &(k, c)) in sub.iter().enumerate() {
                acc += c as u64;
                if acc > median {
                    return (base + i + 1, k);
                }
            }
            unreachable!("weighted median boundary outside range");
        }
        // Median-of-3 pivot on unique keys: strictly above the range min,
        // so the left side is never empty and every round shrinks.
        let a = kc[0];
        let b = kc[n / 2];
        let c = kc[n - 1];
        let pivot = a.max(b).min(a.min(b).max(c));
        #[cfg(target_arch = "x86_64")]
        let (nleft, wl) = if has_avx512() {
            unsafe {
                partition_avx512(
                    &mut kc[..n],
                    &mut cc[..n],
                    &mut ko[..n],
                    &mut co[..n],
                    pivot,
                )
            }
        } else {
            partition_scalar(
                &mut kc[..n],
                &mut cc[..n],
                &mut ko[..n],
                &mut co[..n],
                pivot,
            )
        };
        #[cfg(not(target_arch = "x86_64"))]
        let (nleft, wl) = partition_scalar(
            &mut kc[..n],
            &mut cc[..n],
            &mut ko[..n],
            &mut co[..n],
            pivot,
        );
        if acc + wl > median {
            // left side lives in the other buffer
            n = nleft;
            in_cur = !in_cur;
        } else {
            // right side compacted in place in the current buffer
            acc += wl;
            base += nleft;
            n -= nleft;
        }
    }
}

/// Variance median cut over exact colors. Returns at most `max_colors`.
pub fn median_cut(mut entries: Vec<(u32, u32)>, max_colors: usize) -> Vec<[u8; 3]> {
    if entries.is_empty() {
        return vec![[0, 0, 0]];
    }
    if entries.len() <= max_colors {
        // Fewer distinct colors than slots: exact (lossless) palette.
        return entries
            .iter()
            .map(|&(c, _)| [(c >> 16) as u8, (c >> 8) as u8, c as u8])
            .collect();
    }
    let cv = LabConverter::new();
    // Canonicalize the entries order (histogram layout depends on merge
    // order): boxes are only partitioned below, never sorted, so this
    // initial srgb order is what makes every later float sum — and thus
    // the palette — independent of thread scheduling. (A no-op-cost sort
    // for callers that already sorted, e.g. the exact histogram path.)
    if entries.len() > 16384 {
        entries.par_sort_unstable();
    } else {
        entries.sort_unstable();
    }
    let mut bins: Vec<HBin> = entries
        .par_iter()
        .map(|&(c, n)| HBin {
            count: n,
            srgb: c,
            lab: cv.srgb_to_oklab((c >> 16) as u8, (c >> 8) as u8, c as u8),
        })
        .collect();

    let n = bins.len();
    let mut sel_keys: Vec<u64> = vec![0; n];
    let mut sel_counts: Vec<u32> = vec![0; n];
    let mut tmp_keys: Vec<u64> = vec![0; n];
    let mut tmp_counts: Vec<u32> = vec![0; n];
    let mut part_scratch: Vec<HBin> = vec![bins[0]; n];
    let mut boxes: Vec<Box_> = vec![make_box(&bins, 0, n)];
    while boxes.len() < max_colors {
        let mut best: Option<usize> = None;
        for (i, b) in boxes.iter().enumerate() {
            if b.len > 1 && (best.is_none() || b.cut_score > boxes[best.unwrap()].cut_score) {
                best = Some(i);
            }
        }
        let Some(bi) = best else { break };
        let (start, len, axis, total) = {
            let b = &boxes[bi];
            (b.start, b.len, b.cut_axis, b.count)
        };
        let slice = &mut bins[start..start + len];
        // Count-weighted median split (>=1 color on each side), located by
        // selection over packed keys instead of sorting the box. The srgb
        // tiebreak in the key gives a total order, so the split set is
        // exactly what the old full sort produced.
        let median = total.div_ceil(2);
        for ((k, c), b) in sel_keys[..len]
            .iter_mut()
            .zip(&mut sel_counts[..len])
            .zip(slice.iter())
        {
            *k = axis_key(b, axis);
            *c = b.count;
        }
        let (s, kbound) = weighted_split(
            &mut sel_keys[..len],
            &mut sel_counts[..len],
            &mut tmp_keys[..len],
            &mut tmp_counts[..len],
            median,
        );
        let (split, kbound) = if s >= len {
            // Boundary would be the whole box (its max-key color holds half
            // the weight): clamp to len-1 like the sorted scan did, i.e.
            // split just below the maximum key (= at the second-largest).
            let mut k1 = 0u64;
            let mut k2 = 0u64;
            for b in slice.iter() {
                let k = axis_key(b, axis);
                if k > k1 {
                    k2 = k1;
                    k1 = k;
                } else if k > k2 {
                    k2 = k;
                }
            }
            (len - 1, k2)
        } else {
            (s, kbound)
        };
        // Stable partition around the boundary key: both halves keep the
        // canonical srgb order, so downstream sums stay deterministic.
        let scratch = &mut part_scratch[..len];
        let mut lo = 0usize;
        let mut hi = split;
        for b in slice.iter() {
            let left = axis_key(b, axis) <= kbound;
            let dst = if left { lo } else { hi };
            scratch[dst] = *b;
            lo += left as usize;
            hi += !left as usize;
        }
        debug_assert_eq!(lo, split);
        slice.copy_from_slice(scratch);
        let right = make_box(&bins, start + split, len - split);
        boxes[bi] = make_box(&bins, start, split);
        boxes.push(right);
    }

    boxes
        .iter()
        .map(|bx| {
            let slice = &bins[bx.start..bx.start + bx.len];
            // A box that collapsed to one distinct color must reproduce it
            // exactly — the f32 Lab->sRGB roundtrip can drift by ±1, and in
            // large flat regions that off-by-one turns into dither speckle.
            if slice.len() == 1 {
                let c = slice[0].srgb;
                return [(c >> 16) as u8, (c >> 8) as u8, c as u8];
            }
            let mut count = 0u64;
            let mut sum = [0f64; 3];
            let mut dominant = &slice[0];
            for b in slice {
                count += b.count as u64;
                if b.count > dominant.count {
                    dominant = b;
                }
                for (s, &l) in sum.iter_mut().zip(&b.lab) {
                    *s += b.count as f64 * l as f64;
                }
            }
            // Same reasoning when one color overwhelms the box: the weighted
            // average is that color, so snap to its exact sRGB.
            if dominant.count as u64 * 100 >= count * 99 {
                let c = dominant.srgb;
                return [(c >> 16) as u8, (c >> 8) as u8, c as u8];
            }
            oklab_to_srgb([
                (sum[0] / count as f64) as f32,
                (sum[1] / count as f64) as f32,
                (sum[2] / count as f64) as f32,
            ])
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Exact nearest-color lookup (OkLab metric)

/// Marker byte for multi-candidate cells: appears as the low byte of a
/// multi-candidate `direct[]` entry and as the candidate-list terminator.
/// Palette indices never reach it (palettes cap at 255 colors).
pub const MULTI: u8 = 0xFF;

/// Per-grid-cell candidate lists. `lookup` returns the palette index whose
/// OkLab distance to the query color is minimal; for most cells there is a
/// single candidate and the search collapses to one load.
pub struct NearestMap {
    /// Per-cell entry. Single-candidate cells (the fast path) pack
    /// `r<<24 | g<<16 | b<<8 | idx`, so the caller gets the palette color
    /// with the same load as the index (no dependent colors[] fetch).
    /// Multi-candidate cells pack `offset<<8 | 0xFF`: the low byte can't
    /// collide with a real entry (palettes cap at 255 colors, so the index
    /// byte never reaches 0xFF), and the offset points straight at the
    /// cell's 0xFF-terminated list in `cands` — the whole resolve path
    /// touches one table instead of bouncing through a separate starts[]
    /// array.
    direct: Vec<u32>,
    /// Concatenated candidate lists for multi-candidate cells, each list
    /// terminated by 0xFF.
    cands: Vec<u8>,
    /// packed 24-bit sRGB per palette entry, for re-packing resolve results
    pal_rgb: Vec<u32>,
    pal_lab: Vec<[f32; 3]>,
    cv: LabConverter,
    /// Mean candidates per cell before interning — perf diagnostic.
    avg_cands: f32,
    /// `rgb << 8 | idx` per palette index, padded to 256.
    pal_packed: Box<[u32; 256]>,
}

/// Two-level candidate build geometry: `SUPER_BITS` bits per channel at
/// the coarse level (16^3 super-cells, each spanning a 16-wide RGB box =
/// 4x4x4 of the 4-wide grid cells the lookup indexes).
const SUPER_BITS: u32 = 4;
const SUPER_MASK: usize = (1 << SUPER_BITS) - 1;
const SUPER_SIZE: usize = 1 << (3 * SUPER_BITS);
/// Channel shift from a super-cell index to its base RGB value.
const SUPER_SHIFT: u32 = 8 - SUPER_BITS;
/// RGB values spanned by a super-cell along one axis.
const SUPER_SPAN: u8 = 1 << SUPER_SHIFT;
/// Grid cells per super-cell along one axis, and in total.
const SUB_BITS: u32 = GRID_BITS - SUPER_BITS;

/// Base RGB corner of a grid cell in super-cell-major order: `m`'s high
/// bits pick the super-cell, its low `3 * SUB_BITS` bits the child.
#[inline]
fn cell_base(m: usize) -> (u8, u8, u8) {
    let (sr, sg, sb) = super_base(m >> (3 * SUB_BITS));
    let sub = m & ((1 << (3 * SUB_BITS)) - 1);
    (
        sr + ((((sub >> (2 * SUB_BITS)) as u8) & 3) << 2),
        sg + ((((sub >> SUB_BITS) as u8) & 3) << 2),
        sb + (((sub as u8) & 3) << 2),
    )
}

/// Grid key (the lookup index) of a super-cell-major cell index.
#[inline]
fn cell_key(m: usize) -> usize {
    let (r, g, b) = cell_base(m);
    grid_key(r, g, b)
}

/// Base RGB corner of a super-cell index.
#[inline]
fn super_base(skey: usize) -> (u8, u8, u8) {
    (
        (((skey >> (2 * SUPER_BITS)) & SUPER_MASK) as u8) << SUPER_SHIFT,
        (((skey >> SUPER_BITS) & SUPER_MASK) as u8) << SUPER_SHIFT,
        ((skey & SUPER_MASK) as u8) << SUPER_SHIFT,
    )
}

/// Candidate list for the RGB box `[base, base+span)` against the colors in
/// `soa` (`n` real entries; `map` turns a position in `soa` into the global
/// palette index to emit).
///
/// The reference point q is the box's integer center. Every integer color p
/// in the box satisfies dist(p, q) <= rmax (the max over the 8 corners), so
/// the true nearest c for any such p satisfies
/// dist(c, q) <= dmin(q) + 2*rmax by the triangle inequality — every color
/// inside that bound is kept, which makes the argmin over the list exact
/// for every color in the box.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn cell_candidates<const SPAN: u8>(
    cv: &LabConverter,
    level: fearless_simd::Level,
    soa: &crate::simdops::PalSoa,
    dists: &mut [f32],
    n: usize,
    rb: u8,
    gb: u8,
    bb: u8,
    map: impl Fn(usize) -> u8,
) -> Vec<u8> {
    const { assert!(SPAN.is_power_of_two()) };
    let half = SPAN >> 1;
    let hi = SPAN - 1;
    let q = cv.srgb_to_oklab(rb + half, gb + half, bb + half);
    // All 8 corners in SIMD lanes (slightly inflated to stay an upper
    // bound): candidate lists built from it are supersets of the exact
    // lists, so lookups still return the true nearest.
    let mut lr = [0f32; 8];
    let mut lg = [0f32; 8];
    let mut lb = [0f32; 8];
    for corner in 0..8 {
        lr[corner] = cv.linear(rb + if corner & 1 != 0 { hi } else { 0 });
        lg[corner] = cv.linear(gb + if corner & 2 != 0 { hi } else { 0 });
        lb[corner] = cv.linear(bb + if corner & 4 != 0 { hi } else { 0 });
    }
    let rmax = crate::simdops::corner_rmax2(level, &lr, &lg, &lb, q).sqrt();
    // one SIMD distance pass, buffered, shared by the dmin scan and the
    // candidate filter
    let dmin2 = crate::simdops::cell_distances(level, soa, q, dists);
    let bound = dmin2.sqrt() + 2.0 * rmax + 1e-6;
    let bound2 = bound * bound;
    // A palette that clusters tightly in Lab admits dozens of candidates
    // per cell, and growing 262144 vectors from empty is then a million
    // reallocations. Long scans count first — branchless, and `dists` is
    // still in L1 — and allocate once. Short scans, which is every cell
    // on ordinary content, keep growing from empty: two or three pushes
    // never reach a second allocation anyway.
    let mut list = if n <= 16 {
        Vec::new()
    } else {
        Vec::with_capacity(dists[..n].iter().filter(|&&d| d <= bound2).count())
    };
    for (i, &d) in dists[..n].iter().enumerate() {
        if d <= bound2 {
            list.push(map(i));
        }
    }
    list
}

/// A packed lookup result: `r<<24 | g<<16 | b<<8 | idx`.
pub type PackedNearest = u32;

/// FxHash: the multiply-xor-fold rustc uses. SipHash's quality buys
/// nothing for interning candidate lists — the keys are short byte
/// strings compared in full on every hit — and its cost shows up
/// directly, since a degenerate palette hashes megabytes of them.
#[derive(Default)]
struct FxHasher(u64);

type BuildFx = std::hash::BuildHasherDefault<FxHasher>;

impl FxHasher {
    #[inline(always)]
    fn add(&mut self, w: u64) {
        const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(SEED);
    }
}

impl std::hash::Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, rest) = bytes.as_chunks::<8>();
        for c in chunks {
            self.add(u64::from_le_bytes(*c));
        }
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(buf));
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

impl NearestMap {
    pub fn build(colors: &[[u8; 3]]) -> Self {
        let cv = LabConverter::new();
        let pal_lab: Vec<[f32; 3]> = colors
            .iter()
            .map(|c| cv.srgb_to_oklab(c[0], c[1], c[2]))
            .collect();

        // For each cell: reference point q = the cell's integer center
        // (base+2). Any integer color p in the cell satisfies
        // dist(p, q) <= rmax (max over the 8 integer corners). The true
        // nearest for p then satisfies dist(c, q) <= dmin(q) + 2*rmax, so
        // collecting all palette colors within that bound makes the argmin
        // over candidates exact for every p in the cell.
        //
        // Doing that against the whole palette for all 262144 cells is
        // 67M distance evaluations. Instead the same bound is applied
        // twice: once over 16^3 super-cells (each covering 4x4x4 grid
        // cells) against the full palette, then per grid cell against
        // only its super-cell's candidates. The super-cell list is a
        // superset of every child's - it is built from a strictly larger
        // box around a point of the same lattice - so the result is
        // identical to the flat build, at a fraction of the work.
        let soa = crate::simdops::PalSoa::new(&pal_lab);
        let level = fearless_simd::Level::new();
        let npal = pal_lab.len();
        // A palette that forms one tight cluster in Lab (few distinct
        // source colors) admits nearly every color for every distant
        // super-cell, so the coarse level prunes nothing and its per-cell
        // scan arrays are just duplicated data pushed out of L1. Probe a
        // spread-out sample of super-cells first and skip the whole coarse
        // level when it would not pay.
        let probe: usize = {
            let cv = LabConverter::new();
            let mut dists = vec![0f32; soa.l.len()];
            // a 4x4x4 lattice of super-cells spread through the cube (a
            // plain stride would sample one face of it)
            let step = 1 << (SUPER_BITS - 2);
            (0..64)
                .map(|i| {
                    let sr = ((i >> 4) * step + step / 2) as u8;
                    let sg = (((i >> 2) & 3) * step + step / 2) as u8;
                    let sb = ((i & 3) * step + step / 2) as u8;
                    let (sr, sg, sb) = (sr << SUPER_SHIFT, sg << SUPER_SHIFT, sb << SUPER_SHIFT);
                    cell_candidates::<SUPER_SPAN>(
                        &cv,
                        level,
                        &soa,
                        &mut dists,
                        npal,
                        sr,
                        sg,
                        sb,
                        |i| i as u8,
                    )
                    .len()
                })
                .sum()
        };
        let pruned = probe * 4 <= npal * 64 * 3;

        // Pass A: one candidate list per super-cell, against the full
        // palette, with the surviving colors compacted into their own SoA
        // so the children scan a short contiguous array.
        let supers: Vec<(Vec<u8>, crate::simdops::PalSoa)> = if pruned {
            (0..SUPER_SIZE)
                .into_par_iter()
                .map_init(
                    || (LabConverter::new(), vec![0f32; soa.l.len()]),
                    |(cv, dists), skey| {
                        let (sr, sg, sb) = super_base(skey);
                        let list = cell_candidates::<SUPER_SPAN>(
                            cv,
                            level,
                            &soa,
                            dists,
                            npal,
                            sr,
                            sg,
                            sb,
                            |i| i as u8,
                        );
                        let lab: Vec<[f32; 3]> =
                            list.iter().map(|&i| pal_lab[i as usize]).collect();
                        let sub = crate::simdops::PalSoa::new(&lab);
                        (list, sub)
                    },
                )
                .collect()
        } else {
            Vec::new()
        };
        let widest = supers.iter().map(|(_, s)| s.l.len()).max().unwrap_or(0);
        // Pass B: each grid cell filters only its super-cell's candidates.
        // Cells are visited in super-cell-major order (`m`), so a worker
        // runs all 64 children of one super-cell back to back against the
        // same short scan array instead of cycling through 16 of them; in
        // grid-key order the innermost channel crosses a super-cell
        // boundary every 4 cells. `cell_key` puts the result back.
        let cell_lists: Vec<Vec<u8>> = (0..GRID_SIZE)
            .into_par_iter()
            .map_init(
                || {
                    let n = if pruned { widest } else { soa.l.len() };
                    (LabConverter::new(), vec![0f32; n])
                },
                |(cv, dists), m| {
                    let (rb, gb, bb) = cell_base(m);
                    if !pruned {
                        return cell_candidates::<4>(
                            cv,
                            level,
                            &soa,
                            dists,
                            npal,
                            rb,
                            gb,
                            bb,
                            |i| i as u8,
                        );
                    }
                    let (parent, psoa) = &supers[m >> (3 * SUB_BITS)];
                    cell_candidates::<4>(cv, level, psoa, dists, parent.len(), rb, gb, bb, |i| {
                        parent[i]
                    })
                },
            )
            .collect();

        let pal_rgb: Vec<u32> = colors
            .iter()
            .map(|c| ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32)
            .collect();
        let mut direct = vec![0u32; GRID_SIZE];
        let mut cands = Vec::new();
        // Long lists are interned: when the palette is a tight cluster in
        // Lab (few distinct source colors), the triangle bound admits
        // nearly every color for the vast majority of cells, and those
        // cells all end up with the *same* list. Storing one copy keeps
        // `cands` small (it would otherwise reach 67MB and blow the
        // 24-bit offset packing) and keeps the shared list hot in cache.
        // Short lists — the overwhelming majority on real content — skip
        // the hashing entirely.
        const INTERN_MIN: usize = 16;
        let mut interned: std::collections::HashMap<&[u8], u32, BuildFx> = Default::default();
        // Neighbouring cells usually carry the identical list, so check
        // the last one first and skip the hash entirely on a hit.
        let mut last: (&[u8], u32) = (&[], 0);
        for (m, l) in cell_lists.iter().enumerate() {
            direct[cell_key(m)] = if l.len() == 1 {
                (pal_rgb[l[0] as usize] << 8) | l[0] as u32
            } else {
                let off = if !last.0.is_empty() && last.0 == l.as_slice() {
                    last.1
                } else {
                    match interned.get(l.as_slice()) {
                        Some(&off) => off,
                        None => {
                            let off = cands.len() as u32;
                            // 24-bit offset: with interning the total
                            // stays far below 16M, but guard the packing
                            // invariant
                            assert!(off < (1 << 24), "candidate lists exceed 24-bit offsets");
                            cands.extend_from_slice(l);
                            cands.push(MULTI); // terminator
                            if l.len() >= INTERN_MIN {
                                interned.insert(l.as_slice(), off);
                            }
                            off
                        }
                    }
                };
                if l.len() >= INTERN_MIN {
                    last = (l.as_slice(), off);
                }
                (off << 8) | MULTI as u32
            };
        }
        drop(interned);
        let mut pal_packed = Box::new([0u32; 256]);
        for (i, &rgb) in pal_rgb.iter().enumerate() {
            pal_packed[i] = (rgb << 8) | i as u32;
        }
        let total: usize = cell_lists.iter().map(Vec::len).sum();
        NearestMap {
            direct,
            cands,
            pal_rgb,
            pal_lab,
            cv,
            avg_cands: total as f32 / GRID_SIZE as f32,
            pal_packed,
        }
    }

    /// Average candidates per cell — perf diagnostic.
    pub fn avg_candidates(&self) -> f32 {
        self.avg_cands
    }

    /// Uncached lookup (tests; `lookup_packed` is the hot path).
    #[cfg_attr(not(test), allow(dead_code))]
    #[inline(always)]
    pub fn lookup(&self, r: u8, g: u8, b: u8) -> u8 {
        let key = grid_key(r, g, b);
        let d = self.direct[key];
        if (d & 0xFF) != MULTI as u32 {
            return d as u8;
        }
        self.resolve_off((d >> 8) as usize, r, g, b)
    }

    /// Nearest palette entry as a packed `rgb<<8 | idx` word, memoizing
    /// multi-candidate resolutions in a per-thread direct-mapped cache —
    /// dithered content repeats adjusted colors heavily, and this skips
    /// the Lab conversion for repeats. Returning the palette color with
    /// the index keeps the caller's error math off a dependent load.
    #[inline(always)]
    pub fn lookup_packed(&self, cache: &mut IdxCache, r: u8, g: u8, b: u8) -> PackedNearest {
        let key = grid_key(r, g, b);
        let d = self.direct[key];
        if (d & 0xFF) != MULTI as u32 {
            return d;
        }
        self.lookup_slow(cache, d >> 8, r, g, b)
    }

    /// `rgb << 8 | idx` for every palette index, padded to 256 so a byte
    /// index addresses it in range.
    pub fn packed_palette(&self) -> &[u32; 256] {
        &self.pal_packed
    }

    /// Probe the memo cache for a colour, returning the packed nearest
    /// entry on a hit.
    ///
    /// The staged quantizer probes this *before* the grid table. 80-95% of
    /// pixels hit — a frame's distinct colours are far fewer than its
    /// pixels — and on a hit the 1MB `direct` table is never touched at
    /// all. Only a miss pays the grid lookup, and it is the same cost it
    /// always was. (The old order paid both: the grid load told the caller
    /// the cell was multi-candidate, and only then did the cache probe
    /// happen, so the common pixel took two random loads instead of one.)
    #[inline(always)]
    pub fn cache_probe(&self, cache: &IdxCache, color: u32) -> Option<PackedNearest> {
        let e = cache.slots[cache_slot(color)];
        // the e != MAX guard keeps the empty sentinel (whose tag bits read
        // as 0xFFFFFF) from false-hitting on white; a real white entry has
        // an index byte below 0xFF and never equals MAX
        ((e >> 40) == color as u64 && e != u64::MAX).then_some(e as u32)
    }

    /// Resolve a colour the cache missed on: grid lookup, candidate scan
    /// if the cell holds more than one, then insert. `key` is the colour's
    /// grid key, which the staged pipeline already has in hand.
    #[inline]
    pub fn resolve_keyed(&self, cache: &mut IdxCache, key: u32, color: u32) -> PackedNearest {
        let (r, g, b) = ((color >> 16) as u8, (color >> 8) as u8, color as u8);
        let d = self.direct[key as usize];
        let packed = if (d & 0xFF) != MULTI as u32 {
            d
        } else {
            let idx = self.resolve_off((d >> 8) as usize, r, g, b);
            (self.pal_rgb[idx as usize] << 8) | idx as u32
        };
        cache.slots[cache_slot(color)] = ((color as u64) << 40) | packed as u64;
        packed
    }

    /// Prefetch a colour's memo-cache slot (staged loops run a fixed
    /// distance ahead of themselves).
    #[inline(always)]
    pub fn prefetch_color_slot(&self, cache: &IdxCache, color: u32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            _mm_prefetch(
                cache.slots.as_ptr().add(cache_slot(color)) as *const i8,
                _MM_HINT_T0,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (cache, color);
        }
    }

    /// The multi-candidate path of `lookup_packed`, for callers that
    /// already saw `direct_lookup` return a multi-candidate entry on this
    /// color's cell; `off` is that entry's high 24 bits (the candidate
    /// list offset), so no further cell metadata load is needed.
    #[inline]
    pub fn lookup_slow(
        &self,
        cache: &mut IdxCache,
        off: u32,
        r: u8,
        g: u8,
        b: u8,
    ) -> PackedNearest {
        let color = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
        let slot = cache_slot(color);
        let e = cache.slots[slot];
        // the e != MAX guard keeps the empty sentinel (whose tag bits read
        // as 0xFFFFFF) from false-hitting on white; a real white entry has
        // an index byte below 0xFF and never equals MAX
        if (e >> 40) == color as u64 && e != u64::MAX {
            return e as u32;
        }
        let idx = self.resolve_off(off as usize, r, g, b);
        let packed = (self.pal_rgb[idx as usize] << 8) | idx as u32;
        cache.slots[slot] = ((color as u64) << 40) | packed as u64;
        packed
    }

    /// Prefetch the fast-path cell for a color a few pixels ahead of the
    /// current one: the direct[] table is 1MB, and on colorful content
    /// the dependent load is what stalls the quantize loops. The raw
    /// (pre-dither) color is close enough to the adjusted one to land on
    /// the right cache line almost always.
    #[inline(always)]
    pub fn prefetch(&self, r: u8, g: u8, b: u8) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let key = grid_key(r, g, b);
            _mm_prefetch(self.direct.as_ptr().add(key) as *const i8, _MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (r, g, b);
        }
    }

    /// Prefetch the fast-path cell for an already-computed grid key (the
    /// staged gather loops run a fixed distance ahead of themselves).
    #[inline(always)]
    pub fn prefetch_key(&self, key: u32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            _mm_prefetch(
                self.direct.as_ptr().add(key as usize) as *const i8,
                _MM_HINT_T0,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = key;
        }
    }

    /// Scan the 0xFF-terminated candidate list at `off` for the OkLab
    /// argmin.
    #[inline(never)]
    fn resolve_off(&self, off: usize, r: u8, g: u8, b: u8) -> u8 {
        let q = self.cv.srgb_to_oklab_fast(r, g, b);
        let mut best = 0u8;
        let mut best_d = f32::MAX;
        for &i in self.cands[off..].iter() {
            if i == MULTI {
                break;
            }
            let d = dist2(&self.pal_lab[i as usize], &q);
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }
}

/// Direct-mapped memo cache for multi-candidate nearest lookups (64K
/// slots). Each u64 packs `query_color<<40 | palette_rgb<<8 | idx`, so a
/// probe touches one cache line and a hit returns the palette color along
/// with the index. Empty slots are u64::MAX (see the sentinel guard in
/// `lookup_packed`).
pub struct IdxCache {
    slots: Vec<u64>,
}

/// Direct-mapped slot for a 24-bit colour.
#[inline(always)]
fn cache_slot(color: u32) -> usize {
    (color.wrapping_mul(0x9E37_79B1) >> 16) as usize
}

impl Default for IdxCache {
    fn default() -> Self {
        IdxCache {
            slots: vec![u64::MAX; 1 << 16],
        }
    }
}

#[inline(always)]
fn dist2(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hist_from_frame(frame: &[u8]) -> Vec<(u32, u32)> {
        let mut h = ColorHist::new();
        accumulate_frame(&mut h, frame, &mut Vec::new());
        h.entries()
    }

    #[test]
    fn exact_palette_for_few_colors() {
        let frame: Vec<u8> = [
            [255u8, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 0, 0, 255],
        ]
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect();
        let mut pal = median_cut(hist_from_frame(&frame), 255);
        pal.sort();
        assert_eq!(pal, vec![[0, 0, 255], [0, 255, 0], [255, 0, 0]]);
    }

    #[test]
    fn adjacent_colors_stay_distinct() {
        // colors 1 apart must each survive into the palette (lossless case)
        let frame: Vec<u8> = [[10u8, 10, 10, 255], [11, 11, 11, 255]]
            .iter()
            .flat_map(|p| p.iter().copied())
            .collect();
        let pal = median_cut(hist_from_frame(&frame), 255);
        assert_eq!(pal.len(), 2);
        let nm = NearestMap::build(&pal);
        assert_ne!(nm.lookup(10, 10, 10), nm.lookup(11, 11, 11));
    }

    #[test]
    fn median_cut_splits_to_max() {
        let mut frame = Vec::new();
        for i in 0..4096u32 {
            let r = (i % 64 * 4) as u8;
            let g = (i / 64 * 4 % 256) as u8;
            frame.extend_from_slice(&[r, g, (i % 251) as u8, 255]);
        }
        let pal = median_cut(hist_from_frame(&frame), 255);
        assert_eq!(pal.len(), 255);
    }

    #[test]
    fn hist_counts_and_growth() {
        let mut h = ColorHist::new();
        for i in 0..200_000u32 {
            h.add(i & 0xFF_FFFF, 1);
        }
        let entries = h.entries();
        assert_eq!(entries.len(), 200_000);
        assert!(entries.iter().all(|&(_, c)| c == 1));
        h.add(5, 3);
        let e5 = h.entries().iter().find(|&&(k, _)| k == 5).unwrap().1;
        assert_eq!(e5, 4);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn partition_impls_agree() {
        if !has_avx512() {
            return;
        }
        let mut x = 7u32;
        let mut rng = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for n in [1usize, 7, 8, 9, 64, 65, 200, 513] {
            let keys: Vec<u64> = (0..n).map(|i| ((rng() as u64) << 20) | i as u64).collect();
            let counts: Vec<u32> = (0..n).map(|_| 1 + rng() % 100).collect();
            let pivot = keys[rng() as usize % n];
            let (mut ka, mut ca) = (keys.clone(), counts.clone());
            let (mut kb, mut cb) = (keys.clone(), counts.clone());
            let mut oa = (vec![0u64; n], vec![0u32; n]);
            let mut ob = (vec![0u64; n], vec![0u32; n]);
            let (la, wa) = partition_scalar(&mut ka, &mut ca, &mut oa.0, &mut oa.1, pivot);
            let (lb, wb) =
                unsafe { partition_avx512(&mut kb, &mut cb, &mut ob.0, &mut ob.1, pivot) };
            assert_eq!((la, wa), (lb, wb), "n {n}");
            assert_eq!(oa.0[..la], ob.0[..lb]);
            assert_eq!(oa.1[..la], ob.1[..lb]);
            assert_eq!(ka[..n - la], kb[..n - lb]);
            assert_eq!(ca[..n - la], cb[..n - lb]);
        }
    }

    #[test]
    fn weighted_split_matches_sorted_scan() {
        let mut x = 42u32;
        let mut rng = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for case in 0..200 {
            let n = 2 + (rng() % 500) as usize;
            let pairs: Vec<(u64, u32)> = (0..n)
                .map(|i| {
                    let heavy = case % 3 == 0 && i == n - 1;
                    let w = if heavy { 1_000_000 } else { 1 + rng() % 50 };
                    ((rng() as u64) << 20 | i as u64, w) // unique keys
                })
                .collect();
            let total: u64 = pairs.iter().map(|&(_, c)| c as u64).sum();
            let median = total.div_ceil(2);
            // reference: sort + prefix scan
            let mut sorted = pairs.clone();
            sorted.sort_unstable();
            let mut acc = 0u64;
            let mut want = n;
            for (i, &(_, c)) in sorted.iter().enumerate() {
                acc += c as u64;
                if acc > median {
                    want = i + 1;
                    break;
                }
            }
            let mut keys: Vec<u64> = pairs.iter().map(|&(k, _)| k).collect();
            let mut counts: Vec<u32> = pairs.iter().map(|&(_, c)| c).collect();
            let mut tkeys = vec![0u64; n];
            let mut tcounts = vec![0u32; n];
            let (s, k) = weighted_split(&mut keys, &mut counts, &mut tkeys, &mut tcounts, median);
            assert_eq!(s, want, "case {case} n {n}");
            if s < n {
                assert_eq!(k, sorted[s - 1].0, "boundary key, case {case}");
            }
        }
    }

    #[test]
    fn fold_is_identity_when_small() {
        let entries = vec![(0x102030u32, 5u32), (0xFFFFFF, 1)];
        assert_eq!(maybe_fold(entries.clone()), entries);
    }

    #[test]
    fn fold_means_and_counts() {
        // More entries than bins forces a fold; verify against a brute-force
        // per-cell weighted mean.
        let mut entries = Vec::new();
        let mut x = 1234567u32;
        let mut rng = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        let mut seen = std::collections::HashSet::new();
        while entries.len() <= FOLD_SIZE {
            let c = rng() & 0xFF_FFFF;
            if seen.insert(c) {
                entries.push((c, 1 + rng() % 1000));
            }
        }
        let folded = maybe_fold(entries.clone());
        assert!(folded.len() <= FOLD_SIZE);
        let total_in: u64 = entries.iter().map(|&(_, n)| n as u64).sum();
        let total_out: u64 = folded.iter().map(|&(_, n)| n as u64).sum();
        assert_eq!(total_in, total_out);
        // reference: one-pass weighted sums per cell
        let mut expect: std::collections::HashMap<usize, (u64, [u64; 3])> =
            std::collections::HashMap::new();
        for &(ec, en) in &entries {
            let (r, g, b) = (
                (ec >> 16) as u64,
                ((ec >> 8) & 255) as u64,
                (ec & 255) as u64,
            );
            let e = expect
                .entry(fold_key(r as u8, g as u8, b as u8))
                .or_default();
            e.0 += en as u64;
            e.1[0] += en as u64 * r;
            e.1[1] += en as u64 * g;
            e.1[2] += en as u64 * b;
        }
        assert_eq!(folded.len(), expect.len());
        for &(c, n) in &folded {
            let (r, g, b) = ((c >> 16) as u8, (c >> 8) as u8, c as u8);
            let (cnt, sum) = expect.remove(&fold_key(r, g, b)).expect("cell");
            assert_eq!(n as u64, cnt);
            assert_eq!(r as u64, (sum[0] + cnt / 2) / cnt);
            assert_eq!(g as u64, (sum[1] + cnt / 2) / cnt);
            assert_eq!(b as u64, (sum[2] + cnt / 2) / cnt);
        }
    }

    #[test]
    fn coarse_accumulate_matches_exact_fold() {
        // Random rows with runs and transparency: binning while
        // accumulating must equal folding the exact histogram afterwards.
        let mut x = 99u32;
        let mut rng = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        let mut rows = Vec::new();
        for _ in 0..64 {
            let mut row = Vec::new();
            let mut i = 0;
            while i < 512 {
                let run = 1 + (rng() % 9) as usize;
                let px = [
                    (rng() % 256) as u8,
                    (rng() % 256) as u8,
                    (rng() % 256) as u8,
                    if rng() % 11 == 0 { 0 } else { 255 },
                ];
                for _ in 0..run.min(512 - i) {
                    row.extend_from_slice(&px);
                }
                i += run;
            }
            rows.push(row);
        }
        let mut hist = ColorHist::new();
        let mut bins = new_fold_bins();
        let mut alpha_exact = false;
        let mut alpha_coarse = false;
        for row in &rows {
            alpha_exact |= accumulate_frame(&mut hist, row, &mut Vec::new());
            alpha_coarse |= accumulate_frame_coarse(&mut bins, row, &mut Vec::new());
        }
        assert_eq!(alpha_exact, alpha_coarse);
        let mut expect = new_fold_bins();
        fold_into_bins(&mut expect, &hist.entries());
        assert_eq!(fold_bins_to_entries(&bins), fold_bins_to_entries(&expect));
    }

    #[test]
    fn nearest_map_is_exact() {
        // Candidate-list lookup must equal brute force (OkLab metric),
        // modulo exact distance ties.
        let cv = LabConverter::new();
        let mut pal = Vec::new();
        let mut x = 987654321u32;
        let mut rng = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for _ in 0..255 {
            pal.push([
                (rng() % 256) as u8,
                (rng() % 256) as u8,
                (rng() % 256) as u8,
            ]);
        }
        let nm = NearestMap::build(&pal);
        let pal_lab: Vec<[f32; 3]> = pal
            .iter()
            .map(|c| cv.srgb_to_oklab(c[0], c[1], c[2]))
            .collect();
        for _ in 0..50_000 {
            let (r, g, b) = (
                (rng() % 256) as u8,
                (rng() % 256) as u8,
                (rng() % 256) as u8,
            );
            let got = nm.lookup(r, g, b);
            let q = cv.srgb_to_oklab(r, g, b);
            let bd = pal_lab
                .iter()
                .map(|pl| dist2(pl, &q))
                .fold(f32::MAX, f32::min);
            let gd = dist2(&pal_lab[got as usize], &q);
            assert!(
                (gd - bd).abs() <= bd * 1e-5 + 1e-12,
                "not exact for {r},{g},{b}: got d2={gd} best d2={bd}"
            );
        }
    }

    #[test]
    fn dark_region_exactness() {
        // OkLab stretches darks; verify candidate bounds hold there too.
        let cv = LabConverter::new();
        let mut pal = Vec::new();
        for i in 0..64u8 {
            pal.push([i, i / 2, i / 3]);
        }
        let nm = NearestMap::build(&pal);
        let pal_lab: Vec<[f32; 3]> = pal
            .iter()
            .map(|c| cv.srgb_to_oklab(c[0], c[1], c[2]))
            .collect();
        for r in 0..40u8 {
            for g in 0..40u8 {
                let got = nm.lookup(r, g, 5);
                let q = cv.srgb_to_oklab(r, g, 5);
                let bd = pal_lab
                    .iter()
                    .map(|pl| dist2(pl, &q))
                    .fold(f32::MAX, f32::min);
                let gd = dist2(&pal_lab[got as usize], &q);
                assert!(
                    (gd - bd).abs() <= bd * 1e-5 + 1e-12,
                    "mismatch at {r},{g},5"
                );
            }
        }
    }
}
