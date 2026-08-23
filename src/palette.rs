//! Global palette generation and exact nearest-color lookup, both operating
//! in OkLab space to match ffmpeg's palettegen/paletteuse (ffmpeg >= 5.x).
//!
//! - Histogram: open-addressed hash keyed by exact 24-bit color, so sources
//!   with few distinct colors get a lossless palette.
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

/// Accumulate one RGBA frame. Pixels with alpha < 128 are skipped; returns
/// true when any were, so the pipeline knows the disposal mode (and thus
/// whether delta encoding is possible) before quantization starts.
/// Run-length batching keeps hash traffic low on flat content.
pub fn accumulate_frame(hist: &mut ColorHist, rgba: &[u8]) -> bool {
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
                hist.add(last, run);
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
        hist.add(last, run);
    }
    has_alpha
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

fn make_box(bins: &[HBin], start: usize, len: usize) -> Box_ {
    let slice = &bins[start..start + len];
    let mut count = 0u64;
    let mut sum = [0f64; 3];
    for b in slice {
        count += b.count as u64;
        for (s, &l) in sum.iter_mut().zip(&b.lab) {
            *s += b.count as f64 * l as f64;
        }
    }
    let mean = [
        sum[0] / count as f64,
        sum[1] / count as f64,
        sum[2] / count as f64,
    ];
    let mut er2 = [0f64; 3];
    for b in slice {
        for c in 0..3 {
            let d = b.lab[c] as f64 - mean[c];
            er2[c] += b.count as f64 * d * d;
        }
    }
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
pub fn median_cut(entries: &[(u32, u32)], max_colors: usize) -> Vec<[u8; 3]> {
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
    // the palette — independent of thread scheduling.
    let mut entries = entries.to_vec();
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

/// Per-grid-cell candidate lists. `lookup` returns the palette index whose
/// OkLab distance to the query color is minimal; for most cells there is a
/// single candidate and the search collapses to one load.
pub struct NearestMap {
    /// single-candidate fast path: palette index, or 0xFF when the cell
    /// needs the candidate list (0xFF is never a palette index — palettes
    /// cap at 255 colors, and the transparent slot is not in `colors`)
    direct: Vec<u8>,
    /// start offset into `cands` for each cell; len GRID_SIZE + 1
    starts: Vec<u32>,
    cands: Vec<u8>,
    pal_lab: Vec<[f32; 3]>,
    cv: LabConverter,
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
        let soa = crate::simdops::PalSoa::new(&pal_lab);
        let level = fearless_simd::Level::new();
        let cell_lists: Vec<Vec<u8>> = (0..GRID_SIZE)
            .into_par_iter()
            .map_init(
                || (LabConverter::new(), vec![0f32; soa.l.len()]),
                |(cv, dists), key| {
                    let rb = (((key >> (2 * GRID_BITS)) & 63) as u8) << 2;
                    let gb = (((key >> GRID_BITS) & 63) as u8) << 2;
                    let bb = ((key & 63) as u8) << 2;
                    let q = cv.srgb_to_oklab(rb + 2, gb + 2, bb + 2);
                    // All 8 corners in SIMD lanes (slightly inflated to
                    // stay an upper bound): candidate lists built from it
                    // are supersets of the exact-rmax lists, so lookups
                    // still return the true nearest.
                    let mut lr = [0f32; 8];
                    let mut lg = [0f32; 8];
                    let mut lb = [0f32; 8];
                    for corner in 0..8 {
                        lr[corner] = cv.linear(rb + if corner & 1 != 0 { 3 } else { 0 });
                        lg[corner] = cv.linear(gb + if corner & 2 != 0 { 3 } else { 0 });
                        lb[corner] = cv.linear(bb + if corner & 4 != 0 { 3 } else { 0 });
                    }
                    let rmax2 = crate::simdops::corner_rmax2(level, &lr, &lg, &lb, q);
                    let rmax = rmax2.sqrt();
                    // one SIMD distance pass, buffered, shared by the dmin
                    // scan and the candidate filter
                    let dmin2 = crate::simdops::cell_distances(level, &soa, q, dists);
                    let bound = dmin2.sqrt() + 2.0 * rmax + 1e-6;
                    let bound2 = bound * bound;
                    let mut list = Vec::new();
                    for (i, &d) in dists[..pal_lab.len()].iter().enumerate() {
                        if d <= bound2 {
                            list.push(i as u8);
                        }
                    }
                    list
                },
            )
            .collect();

        let mut direct = Vec::with_capacity(GRID_SIZE);
        let mut starts = Vec::with_capacity(GRID_SIZE + 1);
        let mut cands = Vec::new();
        let mut off = 0u32;
        for l in &cell_lists {
            direct.push(if l.len() == 1 { l[0] } else { 0xFF });
            starts.push(off);
            cands.extend_from_slice(l);
            off += l.len() as u32;
        }
        starts.push(off);
        NearestMap {
            direct,
            starts,
            cands,
            pal_lab,
            cv,
        }
    }

    /// Average candidates per cell — perf diagnostic.
    pub fn avg_candidates(&self) -> f32 {
        self.cands.len() as f32 / GRID_SIZE as f32
    }

    /// Uncached lookup (tests; `lookup_cached` is the hot path).
    #[cfg_attr(not(test), allow(dead_code))]
    #[inline(always)]
    pub fn lookup(&self, r: u8, g: u8, b: u8) -> u8 {
        let key = grid_key(r, g, b);
        let d = self.direct[key];
        if d != 0xFF {
            return d;
        }
        self.resolve_cell(key, r, g, b)
    }

    /// Like `lookup` but memoizes multi-candidate resolutions in a
    /// per-thread direct-mapped cache — dithered content repeats adjusted
    /// colors heavily, and this skips the Lab conversion for repeats.
    #[inline(always)]
    pub fn lookup_cached(&self, cache: &mut IdxCache, r: u8, g: u8, b: u8) -> u8 {
        let key = grid_key(r, g, b);
        let d = self.direct[key];
        if d != 0xFF {
            return d;
        }
        let color = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
        let slot = (color.wrapping_mul(0x9E37_79B1) >> 18) as usize;
        let e = cache.slots[slot];
        // the e != MAX guard keeps the empty sentinel (color 0xFFFFFF,
        // index 0xFF) from false-hitting on white
        if e >> 8 == color && e != u32::MAX {
            return e as u8;
        }
        let idx = self.resolve_cell(key, r, g, b);
        cache.slots[slot] = (color << 8) | idx as u32;
        idx
    }

    /// Prefetch the fast-path cell for a color a few pixels ahead of the
    /// current one: the direct[] table is 256KB, and on colorful content
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

    #[inline(never)]
    fn resolve_cell(&self, key: usize, r: u8, g: u8, b: u8) -> u8 {
        let s = self.starts[key] as usize;
        let e = self.starts[key + 1] as usize;
        self.resolve(&self.cands[s..e], r, g, b)
    }

    #[inline(always)]
    fn resolve(&self, cands: &[u8], r: u8, g: u8, b: u8) -> u8 {
        let q = self.cv.srgb_to_oklab_fast(r, g, b);
        let mut best = cands[0];
        let mut best_d = f32::MAX;
        for &i in cands {
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
/// slots). Key and value share one u32 (color << 8 | palette index) so a
/// probe touches a single cache line: palettes cap at 255 colors, so a
/// real index never exceeds 254 and u32::MAX can mark empty slots.
pub struct IdxCache {
    slots: Vec<u32>,
}

impl Default for IdxCache {
    fn default() -> Self {
        IdxCache {
            slots: vec![u32::MAX; 1 << 14],
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
        accumulate_frame(&mut h, frame);
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
        let mut pal = median_cut(&hist_from_frame(&frame), 255);
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
        let pal = median_cut(&hist_from_frame(&frame), 255);
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
        let pal = median_cut(&hist_from_frame(&frame), 255);
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
