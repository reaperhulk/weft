//! SIMD kernels (fearless_simd): runtime-dispatched, so the shipped
//! baseline-CPU binaries still use AVX2/NEON where the machine has it.

#[allow(unused_imports)]
use fearless_simd::prelude::*;
use fearless_simd::{f32x16, f32x8, i32x16, i32x8, u32x16, u32x8, u8x16, u8x64, Level, Simd};
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
    // lane bytes in LE order are exactly the RGBA byte layout
    let bytes: fearless_simd::u8x32<S> = px.bitcast();
    bytes.store_slice(out);
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

// ---------------------------------------------------------------------------
// Staged blue-noise quantizer kernels. The mode has no cross-pixel
// dependency, so a row is processed as vector passes over per-pixel
// arrays (keys, packed lookup results, probe colors) with the table
// lookups staged as scalar loops between them — portable lanes for the
// math, plain loads for the gathers (NEON has no gather instruction, and
// staged independent scalar loads pipeline comparably to microcoded x86
// gathers).

/// 16 RGBA pixels reinterpreted as one u32 lane each (LE: r|g<<8|b<<16|a<<24).
#[inline(always)]
fn px16<S: Simd>(simd: S, rgba: &[u8]) -> u32x16<S> {
    u8x64::from_slice(simd, &rgba[..64]).bitcast()
}

/// Grid key per lane: (r>>2)<<12 | (g>>2)<<6 | (b>>2), from packed pixels.
#[inline(always)]
fn keys16<S: Simd>(simd: S, px: u32x16<S>) -> u32x16<S> {
    ((px & u32x16::splat(simd, 0xFC)) << 10u32)
        | (((px >> 8u32) & u32x16::splat(simd, 0xFC)) << 4u32)
        | ((px >> 18u32) & u32x16::splat(simd, 0x3F))
}

#[inline(always)]
fn grid_key_scalar(r: u8, g: u8, b: u8) -> u32 {
    (((r as u32) >> 2) << 12) | (((g as u32) >> 2) << 6) | ((b as u32) >> 2)
}

/// Stage 1: grid key per pixel; returns true if any alpha byte < 128.
pub fn bn_keys(level: Level, rgba: &[u8], keys: &mut [u32]) -> bool {
    fearless_simd::dispatch!(level, simd => bn_keys_impl(simd, rgba, keys))
}

/// Stage 1 with the activity gate fused in: grid keys, alpha presence,
/// and the per-pixel dither attenuation (see `att_scalar`) in one pass
/// over the row, sharing the pixel loads. Semantically identical to
/// `bn_keys` + `bn_activity` run separately.
pub fn bn_keys_att(
    level: Level,
    cur: &[u8],
    prev: &[u8],
    gate: u32,
    keys: &mut [u32],
    att: &mut [u32],
) -> bool {
    fearless_simd::dispatch!(level, simd => bn_keys_att_impl(simd, cur, prev, gate, keys, att))
}

#[inline(always)]
fn bn_keys_att_impl<S: Simd>(
    simd: S,
    cur: &[u8],
    prev: &[u8],
    gate: u32,
    keys: &mut [u32],
    att: &mut [u32],
) -> bool {
    let w = keys.len();
    if w == 0 {
        return false;
    }
    let g64 = gate as i32 + 64;
    // pixel 0 has no left neighbor: scalar, like bn_activity
    att[0] = att_scalar(cur, prev, 0, g64);
    keys[0] = grid_key_scalar(cur[0], cur[1], cur[2]);
    let mut any_alpha = cur[3] < 128;
    let zero = i32x16::splat(simd, 0);
    let full = i32x16::splat(simd, 256);
    let g64v = i32x16::splat(simd, g64);
    let mut amin = i32x16::splat(simd, 255);
    let mut i = 1usize;
    while i + 16 <= w {
        let px = px16(simd, &cur[i * 4..]);
        keys16(simd, px).store_slice(&mut keys[i..i + 16]);
        amin = amin.min((px >> 24u32).bitcast::<i32x16<S>>());
        let pl = px16(simd, &cur[(i - 1) * 4..]);
        let pu = px16(simd, &prev[i * 4..]);
        let act = sad3(simd, px, pl) + sad3(simd, px, pu);
        let a = ((g64v - act) << 2u32).max(zero).min(full);
        a.bitcast::<u32x16<S>>().store_slice(&mut att[i..i + 16]);
        i += 16;
    }
    let arr: [i32; 16] = amin.into();
    any_alpha |= arr.iter().any(|&a| a < 128);
    while i < w {
        let p = &cur[i * 4..i * 4 + 4];
        keys[i] = grid_key_scalar(p[0], p[1], p[2]);
        any_alpha |= p[3] < 128;
        att[i] = att_scalar(cur, prev, i, g64);
        i += 1;
    }
    any_alpha
}

#[inline(always)]
fn bn_keys_impl<S: Simd>(simd: S, rgba: &[u8], keys: &mut [u32]) -> bool {
    let w = keys.len();
    let mut amin = i32x16::splat(simd, 255);
    let mut i = 0usize;
    while i + 16 <= w {
        let px = px16(simd, &rgba[i * 4..]);
        keys16(simd, px).store_slice(&mut keys[i..i + 16]);
        // alpha lanes are <= 255, so a signed min is safe
        amin = amin.min((px >> 24u32).bitcast::<i32x16<S>>());
        i += 16;
    }
    let arr: [i32; 16] = amin.into();
    let mut any_alpha = arr.iter().any(|&a| a < 128);
    while i < w {
        let p = &rgba[i * 4..i * 4 + 4];
        keys[i] = grid_key_scalar(p[0], p[1], p[2]);
        any_alpha |= p[3] < 128;
        i += 1;
    }
    any_alpha
}

/// Stage 4: from packed c1 results, compute per pixel the nonzero-error
/// flag (`ors`, 0 means the pixel matched its palette color exactly), the
/// clamped far-probe color (`c2c`, packed r<<16|g<<8|b), and its grid key.
/// Returns true if any pixel had nonzero error — false lets the caller
/// skip the candidate-2 stages entirely.
pub fn bn_probe(
    level: Level,
    rgba: &[u8],
    pk1: &[u32],
    ors: &mut [u32],
    c2c: &mut [u32],
    keys2: &mut [u32],
) -> bool {
    fearless_simd::dispatch!(level, simd => bn_probe_impl(simd, rgba, pk1, ors, c2c, keys2))
}

#[inline(always)]
fn bn_probe_impl<S: Simd>(
    simd: S,
    rgba: &[u8],
    pk1: &[u32],
    ors: &mut [u32],
    c2c: &mut [u32],
    keys2: &mut [u32],
) -> bool {
    let w = pk1.len();
    let ff = u32x16::splat(simd, 0xFF);
    let zero = i32x16::splat(simd, 0);
    let hi = i32x16::splat(simd, 255);
    let mut oacc = u32x16::splat(simd, 0);
    let mut i = 0usize;
    while i + 16 <= w {
        let px = px16(simd, &rgba[i * 4..]);
        let p1 = u32x16::from_slice(simd, &pk1[i..i + 16]);
        let r: i32x16<S> = (px & ff).bitcast();
        let g: i32x16<S> = ((px >> 8u32) & ff).bitcast();
        let b: i32x16<S> = ((px >> 16u32) & ff).bitcast();
        let er = r - (p1 >> 24u32).bitcast::<i32x16<S>>();
        let eg = g - ((p1 >> 16u32) & ff).bitcast::<i32x16<S>>();
        let eb = b - ((p1 >> 8u32) & ff).bitcast::<i32x16<S>>();
        let o: u32x16<S> = (er | eg | eb).bitcast();
        o.store_slice(&mut ors[i..i + 16]);
        oacc |= o;
        let r2 = (r + er + er).max(zero).min(hi).bitcast::<u32x16<S>>();
        let g2 = (g + eg + eg).max(zero).min(hi).bitcast::<u32x16<S>>();
        let b2 = (b + eb + eb).max(zero).min(hi).bitcast::<u32x16<S>>();
        ((r2 << 16u32) | (g2 << 8u32) | b2).store_slice(&mut c2c[i..i + 16]);
        (((r2 >> 2u32) << 12u32) | ((g2 >> 2u32) << 6u32) | (b2 >> 2u32))
            .store_slice(&mut keys2[i..i + 16]);
        i += 16;
    }
    let arr: [u32; 16] = oacc.into();
    let mut any_err = arr.iter().any(|&o| o != 0);
    while i < w {
        let p = &rgba[i * 4..i * 4 + 4];
        let p1 = pk1[i];
        let er = p[0] as i32 - (p1 >> 24) as i32;
        let eg = p[1] as i32 - ((p1 >> 16) & 0xFF) as i32;
        let eb = p[2] as i32 - ((p1 >> 8) & 0xFF) as i32;
        ors[i] = (er | eg | eb) as u32;
        any_err |= ors[i] != 0;
        let r2 = (p[0] as i32 + 2 * er).clamp(0, 255) as u32;
        let g2 = (p[1] as i32 + 2 * eg).clamp(0, 255) as u32;
        let b2 = (p[2] as i32 + 2 * eb).clamp(0, 255) as u32;
        c2c[i] = (r2 << 16) | (g2 << 8) | b2;
        keys2[i] = grid_key_scalar(r2 as u8, g2 as u8, b2 as u8);
        i += 1;
    }
    any_err
}

/// Summed per-channel absolute difference of two packed-RGBA vectors
/// (alpha excluded). Byte-wise max - min gives every |channel diff| in
/// one subtraction; the alpha byte is masked off and the three remaining
/// bytes of each lane summed (each <= 255, so the 32-bit sums can't
/// carry into one another).
#[inline(always)]
fn sad3<S: Simd>(simd: S, a: u32x16<S>, b: u32x16<S>) -> i32x16<S> {
    let ab = a.bitcast::<u8x64<S>>();
    let bb = b.bitcast::<u8x64<S>>();
    let d = (ab.max(bb) - ab.min(bb)).bitcast::<u32x16<S>>();
    let ff = u32x16::splat(simd, 0xFF);
    ((d & ff) + ((d >> 8u32) & ff) + ((d >> 16u32) & ff)).bitcast()
}

/// Scalar `bn_activity` for one pixel (vector-loop edges and tails; also
/// the reference the parity tests check the lanes against). `g64` is
/// `gate + 64`.
#[inline(always)]
pub fn att_scalar(cur: &[u8], prev: &[u8], i: usize, g64: i32) -> u32 {
    let li = i.saturating_sub(1);
    let p = &cur[i * 4..i * 4 + 4];
    let l = &cur[li * 4..li * 4 + 4];
    let u = &prev[i * 4..i * 4 + 4];
    let mut act = 0i32;
    for c in 0..3 {
        act += (p[c] as i32 - l[c] as i32).abs() + (p[c] as i32 - u[c] as i32).abs();
    }
    ((g64 - act) << 2).clamp(0, 256) as u32
}

/// Stage 7: the two-candidate threshold pick. `mrow32` is the row's
/// blue-noise threshold line as u32 (64 entries, repeating every 64
/// pixels — rows start at x = 0, so chunks of 16 stay aligned to it).
/// `att` is the per-pixel dither attenuation in 0..=256 (256 = the
/// ungated pick; 0 always keeps c1). For pixels with zero error, equal
/// candidates, or zero attenuation the math degenerates to picking c1,
/// so no lane needs special casing (garbage pk2 lanes on exact or fully
/// attenuated pixels produce den >= 0 and num * att == 0, which always
/// keeps c1).
pub fn bn_threshold(
    level: Level,
    rgba: &[u8],
    pk1: &[u32],
    pk2: &[u32],
    mrow32: &[u32; 64],
    att: &[u32],
    out: &mut [u8],
) {
    fearless_simd::dispatch!(level, simd => bn_threshold_impl(simd, rgba, pk1, pk2, mrow32, att, out))
}

/// The per-16-lane pick of stage 7: palette index per pixel as u32 lanes
/// (only the low byte is nonzero).
#[inline(always)]
fn bn_pick16<S: Simd>(
    simd: S,
    rgba: &[u8],
    pk1: &[u32],
    pk2: &[u32],
    m16: &[u32],
    att: &[u32],
) -> u32x16<S> {
    let ff = u32x16::splat(simd, 0xFF);
    let zero = i32x16::splat(simd, 0);
    let px = px16(simd, rgba);
    let p1 = u32x16::from_slice(simd, &pk1[..16]);
    let p2 = u32x16::from_slice(simd, &pk2[..16]);
    let c1r: i32x16<S> = (p1 >> 24u32).bitcast();
    let c1g: i32x16<S> = ((p1 >> 16u32) & ff).bitcast();
    let c1b: i32x16<S> = ((p1 >> 8u32) & ff).bitcast();
    let er = (px & ff).bitcast::<i32x16<S>>() - c1r;
    let eg = ((px >> 8u32) & ff).bitcast::<i32x16<S>>() - c1g;
    let eb = ((px >> 16u32) & ff).bitcast::<i32x16<S>>() - c1b;
    let dr = (p2 >> 24u32).bitcast::<i32x16<S>>() - c1r;
    let dg = ((p2 >> 16u32) & ff).bitcast::<i32x16<S>>() - c1g;
    let db = ((p2 >> 8u32) & ff).bitcast::<i32x16<S>>() - c1b;
    let num = (er * dr + eg * dg + eb * db).max(zero);
    let den = dr * dr + dg * dg + db * db;
    let m: i32x16<S> = u32x16::from_slice(simd, &m16[..16]).bitcast();
    let attv: i32x16<S> = u32x16::from_slice(simd, &att[..16]).bitcast();
    let pick = (m * den).simd_lt(num * attv);
    pick.select(p2 & ff, p1 & ff)
}

#[inline(always)]
fn bn_threshold_impl<S: Simd>(
    simd: S,
    rgba: &[u8],
    pk1: &[u32],
    pk2: &[u32],
    mrow32: &[u32; 64],
    att: &[u32],
    out: &mut [u8],
) {
    let w = out.len();
    let mut i = 0usize;
    // 64 pixels per round: four picks, then two unzip rounds compact the
    // u32 lanes' low bytes into one contiguous 64-byte store (each unzip
    // keeps even-indexed bytes: u32 lanes -> [idx, 0] u16 pairs -> idx
    // bytes in order).
    while i + 64 <= w {
        let k = i & 63; // 0: chunks stay mask-aligned
        let v0 = bn_pick16(
            simd,
            &rgba[i * 4..],
            &pk1[i..],
            &pk2[i..],
            &mrow32[k..],
            &att[i..],
        );
        let v1 = bn_pick16(
            simd,
            &rgba[(i + 16) * 4..],
            &pk1[i + 16..],
            &pk2[i + 16..],
            &mrow32[k + 16..],
            &att[i + 16..],
        );
        let v2 = bn_pick16(
            simd,
            &rgba[(i + 32) * 4..],
            &pk1[i + 32..],
            &pk2[i + 32..],
            &mrow32[k + 32..],
            &att[i + 32..],
        );
        let v3 = bn_pick16(
            simd,
            &rgba[(i + 48) * 4..],
            &pk1[i + 48..],
            &pk2[i + 48..],
            &mrow32[k + 48..],
            &att[i + 48..],
        );
        let t0 = v0.bitcast::<u8x64<S>>().unzip_low(v1.bitcast::<u8x64<S>>());
        let t1 = v2.bitcast::<u8x64<S>>().unzip_low(v3.bitcast::<u8x64<S>>());
        t0.unzip_low(t1).store_slice(&mut out[i..i + 64]);
        i += 64;
    }
    while i + 16 <= w {
        let k = i & 63;
        let v = bn_pick16(
            simd,
            &rgba[i * 4..],
            &pk1[i..],
            &pk2[i..],
            &mrow32[k..],
            &att[i..],
        );
        let arr: [u32; 16] = v.into();
        for (o, x) in out[i..i + 16].iter_mut().zip(arr) {
            *o = x as u8;
        }
        i += 16;
    }
    while i < w {
        let p = &rgba[i * 4..i * 4 + 4];
        let p1 = pk1[i];
        let p2 = pk2[i];
        let c1r = (p1 >> 24) as i32;
        let c1g = ((p1 >> 16) & 0xFF) as i32;
        let c1b = ((p1 >> 8) & 0xFF) as i32;
        let er = p[0] as i32 - c1r;
        let eg = p[1] as i32 - c1g;
        let eb = p[2] as i32 - c1b;
        let dr = (p2 >> 24) as i32 - c1r;
        let dg = ((p2 >> 16) & 0xFF) as i32 - c1g;
        let db = ((p2 >> 8) & 0xFF) as i32 - c1b;
        let num = (er * dr + eg * dg + eb * db).max(0);
        let den = dr * dr + dg * dg + db * db;
        let m = mrow32[i & 63] as i32;
        out[i] = if m * den < num * att[i] as i32 {
            p2 as u8
        } else {
            p1 as u8
        };
        i += 1;
    }
}

/// Fill `out` with the transparency-punched delta of one row (`trans`
/// where cur == prev, else cur) and return how many bytes were punched.
/// Palette indices never collide with `trans` (it's the reserved slot),
/// so the punched count equals the equal-byte count.
pub fn punch_row(level: Level, cur: &[u8], prev: &[u8], trans: u8, out: &mut [u8]) -> usize {
    fearless_simd::dispatch!(level, simd => punch_row_impl(simd, cur, prev, trans, out))
}

#[inline(always)]
fn punch_row_impl<S: Simd>(simd: S, cur: &[u8], prev: &[u8], trans: u8, out: &mut [u8]) -> usize {
    let n = cur.len();
    let tv = u8x16::splat(simd, trans);
    let ones = u8x16::splat(simd, 1);
    let zeros = u8x16::splat(simd, 0);
    let mut count = 0usize;
    let mut i = 0usize;
    while i + 16 <= n {
        // inner loop bounded so the u8 lane counters can't wrap at 256
        let mut acc = zeros;
        let mut blocks = 0u32;
        while i + 16 <= n && blocks < 128 {
            let a = u8x16::from_slice(simd, &cur[i..i + 16]);
            let b = u8x16::from_slice(simd, &prev[i..i + 16]);
            let m = a.simd_eq(b);
            m.select(tv, a).store_slice(&mut out[i..i + 16]);
            acc += m.select(ones, zeros);
            i += 16;
            blocks += 1;
        }
        let arr: [u8; 16] = acc.into();
        count += arr.iter().map(|&v| v as usize).sum::<usize>();
    }
    while i < n {
        if cur[i] == prev[i] {
            out[i] = trans;
            count += 1;
        } else {
            out[i] = cur[i];
        }
        i += 1;
    }
    count
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
        let padded = labs.len().div_ceil(16) * 16;
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
    let ql = f32x16::splat(simd, q[0]);
    let qa = f32x16::splat(simd, q[1]);
    let qb = f32x16::splat(simd, q[2]);
    let mut minv = f32x16::splat(simd, f32::MAX);
    let n = pal.l.len();
    let mut i = 0;
    while i < n {
        let dl = f32x16::from_slice(simd, &pal.l[i..i + 16]) - ql;
        let da = f32x16::from_slice(simd, &pal.a[i..i + 16]) - qa;
        let db = f32x16::from_slice(simd, &pal.b[i..i + 16]) - qb;
        let d = dl * dl + da * da + db * db;
        d.store_slice(&mut dists[i..i + 16]);
        minv = minv.min(d);
        i += 16;
    }
    let arr: [f32; 16] = minv.into();
    arr.iter().fold(f32::MAX, |m, &v| m.min(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_keys_att_lanes_match_scalar() {
        // bn_keys_att's vector lanes must agree with the scalar key and
        // attenuation formulas for every pixel, including the x = 0 edge
        // and the non-multiple-of-16 tail, and must flag alpha the same
        // way bn_keys does.
        let mut x = 0xC0FFEEu32;
        let mut rng = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for w in [1usize, 5, 16, 17, 100, 130] {
            let cur: Vec<u8> = (0..w * 4).map(|_| (rng() & 0xFF) as u8).collect();
            let prev: Vec<u8> = (0..w * 4).map(|_| (rng() & 0xFF) as u8).collect();
            for gate in [1u32, 16, 100] {
                let mut att = vec![0u32; w];
                let mut keys = vec![0u32; w];
                let alpha = bn_keys_att(Level::new(), &cur, &prev, gate, &mut keys, &mut att);
                let mut keys_ref = vec![0u32; w];
                let alpha_ref = bn_keys(Level::new(), &cur, &mut keys_ref);
                assert_eq!(alpha, alpha_ref, "w {w}");
                assert_eq!(keys, keys_ref, "w {w}");
                for (i, &a) in att.iter().enumerate() {
                    let want = att_scalar(&cur, &prev, i, gate as i32 + 64);
                    assert_eq!(a, want, "w {w} gate {gate} i {i}");
                }
            }
        }
    }

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
