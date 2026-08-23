//! SIMD kernels (fearless_simd): runtime-dispatched, so the shipped
//! baseline-CPU binaries still use AVX2/NEON where the machine has it.

#[allow(unused_imports)]
use fearless_simd::prelude::*;
use fearless_simd::{f32x8, i32x8, u32x8, u8x16, Level, Simd};
use std::sync::OnceLock;

/// Runtime-detected SIMD level, cached (feature detection once).
pub fn level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(Level::new)
}

// BT.601 limited-range coefficients, 16.16 fixed point (same constants as
// the scalar path in color.rs — results must stay byte-identical).
const CY: i32 = 76309;
const CRV: i32 = 104597;
const CGU: i32 = -25675;
const CGV: i32 = -53279;
const CBU: i32 = 132201;
const ROUND: i32 = 1 << 15;
const ALPHA: i32 = 0xFF << 24;

/// Convert one row of YUV to RGBA. `cx_shift` = 1 for 4:2:0/4:2:2 chroma
/// (each chroma sample covers two pixels), 0 for 4:4:4. Returns the number
/// of pixels converted; the caller finishes the tail with scalar code.
pub fn convert_row(
    level: Level,
    yrow: &[u8],
    urow: &[u8],
    vrow: &[u8],
    cx_shift: u32,
    out: &mut [u8],
) -> usize {
    fearless_simd::dispatch!(level, simd => convert_row_impl(simd, yrow, urow, vrow, cx_shift, out))
}

#[inline(always)]
fn convert_row_impl<S: Simd>(
    simd: S,
    yrow: &[u8],
    urow: &[u8],
    vrow: &[u8],
    cx_shift: u32,
    out: &mut [u8],
) -> usize {
    let w = yrow.len();
    let mut x = 0usize;
    // u8x16 loads need 16 readable bytes in both luma and chroma rows;
    // the last block near the row end falls back to the scalar tail.
    while x + 16 <= w && (x >> cx_shift) + 16 <= urow.len() {
        let cx = x >> cx_shift;
        let yv = u8x16::from_slice(simd, &yrow[x..x + 16]);
        let (ylo, yhi) = yv.widen();
        let (u0, u1, v0, v1) = if cx_shift == 1 {
            // 8 chroma samples cover 16 pixels: duplicate pairwise
            let (culo, _) = u8x16::from_slice(simd, &urow[cx..cx + 16]).widen();
            let (cvlo, _) = u8x16::from_slice(simd, &vrow[cx..cx + 16]).widen();
            (
                culo.zip_low(culo),
                culo.zip_high(culo),
                cvlo.zip_low(cvlo),
                cvlo.zip_high(cvlo),
            )
        } else {
            let (culo, cuhi) = u8x16::from_slice(simd, &urow[cx..cx + 16]).widen();
            let (cvlo, cvhi) = u8x16::from_slice(simd, &vrow[cx..cx + 16]).widen();
            (culo, cuhi, cvlo, cvhi)
        };
        convert_group(simd, ylo, u0, v0, &mut out[x * 4..x * 4 + 32]);
        convert_group(simd, yhi, u1, v1, &mut out[x * 4 + 32..x * 4 + 64]);
        x += 16;
    }
    x
}

#[inline(always)]
fn convert_group<S: Simd>(
    simd: S,
    y16: fearless_simd::u16x8<S>,
    u16v: fearless_simd::u16x8<S>,
    v16v: fearless_simd::u16x8<S>,
    out: &mut [u8],
) {
    let (ya, yb) = y16.widen();
    let yi: i32x8<S> = ya.combine(yb).bitcast();
    let (ua, ub) = u16v.widen();
    let ui: i32x8<S> = ua.combine(ub).bitcast();
    let (va, vb) = v16v.widen();
    let vi: i32x8<S> = va.combine(vb).bitcast();

    let c = (yi - 16) * CY;
    let d = ui - 128;
    let e = vi - 128;
    let zero = i32x8::splat(simd, 0);
    let hi = i32x8::splat(simd, 255);
    let r = ((c + e * CRV + ROUND) >> 16u32).max(zero).min(hi);
    let g = ((c + d * CGU + e * CGV + ROUND) >> 16u32).max(zero).min(hi);
    let b = ((c + d * CBU + ROUND) >> 16u32).max(zero).min(hi);
    let px = r | (g << 8u32) | (b << 16u32) | ALPHA;
    let mut tmp = [0i32; 8];
    px.store_slice(&mut tmp);
    for (dst, v) in out.as_chunks_mut::<4>().0.iter_mut().zip(tmp) {
        *dst = v.to_le_bytes();
    }
}

/// Vectorized fast cube root for 8 non-negative lanes: shift-series
/// approximation of bits/3 plus the Kahan offset for the seed, then two
/// Halley iterations — a few ULPs of accuracy, like `oklab::cbrt_fast`.
#[inline(always)]
fn cbrt_x8<S: Simd>(simd: S, x: f32x8<S>) -> f32x8<S> {
    let bits: u32x8<S> = x.bitcast();
    // t = bits * (1/4 + 1/16) * (1 + 1/16) * (1 + 1/256) ~= bits / 3
    let t = (bits >> 2u32) + (bits >> 4u32);
    let t = t + (t >> 4u32);
    let t = t + (t >> 8u32);
    let seed = t + u32x8::splat(simd, 709_921_077);
    let mut y: f32x8<S> = seed.bitcast();
    let two = f32x8::splat(simd, 2.0);
    for _ in 0..2 {
        let y3 = y * y * y;
        y = y * (y3 + two * x) / (two * y3 + x);
    }
    y
}

/// Max squared OkLab distance from 8 cell corners (given as linearized
/// sRGB channel values per corner) to the reference point `q`. All eight
/// corners are evaluated in lanes with the vectorized cube root; the
/// result is inflated slightly so bounds built from it remain valid upper
/// bounds despite the cbrt approximation.
pub fn corner_rmax2(level: Level, lr: &[f32; 8], lg: &[f32; 8], lb: &[f32; 8], q: [f32; 3]) -> f32 {
    fearless_simd::dispatch!(level, simd => corner_rmax2_impl(simd, lr, lg, lb, q))
}

#[inline(always)]
fn corner_rmax2_impl<S: Simd>(
    simd: S,
    lr: &[f32; 8],
    lg: &[f32; 8],
    lb: &[f32; 8],
    q: [f32; 3],
) -> f32 {
    let lrv = f32x8::from_slice(simd, lr);
    let lgv = f32x8::from_slice(simd, lg);
    let lbv = f32x8::from_slice(simd, lb);
    let l = lrv * 0.4122214708 + lgv * 0.5363325363 + lbv * 0.0514459929;
    let m = lrv * 0.2119034982 + lgv * 0.6806995451 + lbv * 0.1073969566;
    let s = lrv * 0.0883024619 + lgv * 0.2817188376 + lbv * 0.6299787005;
    let l_ = cbrt_x8(simd, l);
    let m_ = cbrt_x8(simd, m);
    let s_ = cbrt_x8(simd, s);
    let dl = l_ * 0.2104542553 + m_ * 0.7936177850 + s_ * -0.0040720468 - f32x8::splat(simd, q[0]);
    let da = l_ * 1.9779984951 + m_ * -2.4285922050 + s_ * 0.4505937099 - f32x8::splat(simd, q[1]);
    let db = l_ * 0.0259040371 + m_ * 0.7827717662 + s_ * -0.8086757660 - f32x8::splat(simd, q[2]);
    let d = dl * dl + da * da + db * db;
    let arr: [f32; 8] = d.into();
    arr.iter().fold(0f32, |m, &v| m.max(v)) * 1.0002
}

/// Palette OkLab channels in structure-of-arrays form, padded to a
/// multiple of 8 lanes with +inf (padding never wins a min and never
/// passes a <= bound test).
pub struct PalSoa {
    pub l: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl PalSoa {
    pub fn new(labs: &[[f32; 3]]) -> Self {
        let padded = labs.len().div_ceil(8) * 8;
        let mut l = vec![f32::INFINITY; padded];
        let mut a = vec![f32::INFINITY; padded];
        let mut b = vec![f32::INFINITY; padded];
        for (i, lab) in labs.iter().enumerate() {
            l[i] = lab[0];
            a[i] = lab[1];
            b[i] = lab[2];
        }
        PalSoa { l, a, b }
    }
}

/// Fill `dists[i]` with the squared OkLab distance from `q` to palette
/// color i and return the minimum. `dists` must be at least the padded
/// palette length.
pub fn cell_distances(level: Level, pal: &PalSoa, q: [f32; 3], dists: &mut [f32]) -> f32 {
    fearless_simd::dispatch!(level, simd => cell_distances_impl(simd, pal, q, dists))
}

#[inline(always)]
fn cell_distances_impl<S: Simd>(simd: S, pal: &PalSoa, q: [f32; 3], dists: &mut [f32]) -> f32 {
    let ql = f32x8::splat(simd, q[0]);
    let qa = f32x8::splat(simd, q[1]);
    let qb = f32x8::splat(simd, q[2]);
    let mut minv = f32x8::splat(simd, f32::MAX);
    let n = pal.l.len();
    let mut i = 0;
    while i < n {
        let dl = f32x8::from_slice(simd, &pal.l[i..i + 8]) - ql;
        let da = f32x8::from_slice(simd, &pal.a[i..i + 8]) - qa;
        let db = f32x8::from_slice(simd, &pal.b[i..i + 8]) - qb;
        let d = dl * dl + da * da + db * db;
        d.store_slice(&mut dists[i..i + 8]);
        minv = minv.min(d);
        i += 8;
    }
    let arr: [f32; 8] = minv.into();
    arr.iter().fold(f32::MAX, |m, &v| m.min(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_scalar() {
        let labs: Vec<[f32; 3]> = (0..255)
            .map(|i| {
                let f = i as f32;
                [f * 0.001, (f * 0.7).sin() * 0.2, (f * 0.3).cos() * 0.2]
            })
            .collect();
        let pal = PalSoa::new(&labs);
        let level = Level::new();
        let q = [0.4f32, -0.05, 0.11];
        let mut dists = vec![0f32; pal.l.len()];
        let dmin = cell_distances(level, &pal, q, &mut dists);
        let mut smin = f32::MAX;
        for (i, lab) in labs.iter().enumerate() {
            let d = (lab[0] - q[0]).powi(2) + (lab[1] - q[1]).powi(2) + (lab[2] - q[2]).powi(2);
            assert!((dists[i] - d).abs() <= d * 1e-6 + 1e-12, "idx {i}");
            smin = smin.min(d);
        }
        assert!((dmin - smin).abs() <= smin * 1e-6 + 1e-12);
        // padding lanes must never look attractive
        for &d in &dists[labs.len()..] {
            assert!(d.is_infinite());
        }
    }
}
