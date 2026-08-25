//! SIMD kernels (fearless_simd): runtime-dispatched, so the shipped
//! baseline-CPU binaries still use AVX2/NEON where the machine has it.

#[allow(unused_imports)]
use fearless_simd::prelude::*;
use fearless_simd::{
    f32x16, f32x8, i16x32, i32x16, i32x8, mask16x32, mask32x16, u16x32, u32x16, u32x8, u8x16,
    u8x64, Level, Simd,
};
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

/// Convert one YUV row to RGBA while also emitting the 6-bit/channel grid
/// key consumed by the staged blue-noise quantizer. The RGB channels are
/// already live as SIMD lanes here, so producing keys avoids rereading and
/// unpacking the completed RGBA row in a second pass.
pub fn convert_row_with_keys(
    level: Level,
    yrow: &[u8],
    urow: &[u8],
    vrow: &[u8],
    cx_shift: u32,
    out: &mut [u8],
    keys: &mut [u32],
) -> usize {
    fearless_simd::dispatch!(level, simd => convert_row_with_keys_impl(
        simd, yrow, urow, vrow, cx_shift, out, keys
    ))
}

/// Convert one YUV row directly to canonical `0xRRGGBB` keys for the
/// histogram pass, which does not otherwise need an RGBA materialization.
pub fn convert_row_to_rgb_keys(
    level: Level,
    yrow: &[u8],
    urow: &[u8],
    vrow: &[u8],
    cx_shift: u32,
    keys: &mut [u32],
) -> usize {
    fearless_simd::dispatch!(level, simd => convert_row_to_rgb_keys_impl(
        simd, yrow, urow, vrow, cx_shift, keys
    ))
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
fn convert_row_with_keys_impl<S: Simd>(
    simd: S,
    yrow: &[u8],
    urow: &[u8],
    vrow: &[u8],
    cx_shift: u32,
    out: &mut [u8],
    keys: &mut [u32],
) -> usize {
    let w = yrow.len();
    let mut x = 0usize;
    while x + 16 <= w && (x >> cx_shift) + 16 <= urow.len() {
        let cx = x >> cx_shift;
        let yv = u8x16::from_slice(simd, &yrow[x..x + 16]);
        let (ylo, yhi) = yv.widen();
        let (u0, u1, v0, v1) = chroma_groups(simd, urow, vrow, cx, cx_shift);
        convert_group_with_keys(
            simd,
            ylo,
            u0,
            v0,
            &mut out[x * 4..x * 4 + 32],
            &mut keys[x..x + 8],
        );
        convert_group_with_keys(
            simd,
            yhi,
            u1,
            v1,
            &mut out[x * 4 + 32..x * 4 + 64],
            &mut keys[x + 8..x + 16],
        );
        x += 16;
    }
    x
}

#[inline(always)]
fn convert_row_to_rgb_keys_impl<S: Simd>(
    simd: S,
    yrow: &[u8],
    urow: &[u8],
    vrow: &[u8],
    cx_shift: u32,
    keys: &mut [u32],
) -> usize {
    let w = yrow.len();
    let mut x = 0usize;
    while x + 16 <= w && (x >> cx_shift) + 16 <= urow.len() {
        let cx = x >> cx_shift;
        let yv = u8x16::from_slice(simd, &yrow[x..x + 16]);
        let (ylo, yhi) = yv.widen();
        let (u0, u1, v0, v1) = chroma_groups(simd, urow, vrow, cx, cx_shift);
        convert_group_to_rgb_keys(simd, ylo, u0, v0, &mut keys[x..x + 8]);
        convert_group_to_rgb_keys(simd, yhi, u1, v1, &mut keys[x + 8..x + 16]);
        x += 16;
    }
    x
}

#[inline(always)]
fn chroma_groups<S: Simd>(
    simd: S,
    urow: &[u8],
    vrow: &[u8],
    cx: usize,
    cx_shift: u32,
) -> (
    fearless_simd::u16x8<S>,
    fearless_simd::u16x8<S>,
    fearless_simd::u16x8<S>,
    fearless_simd::u16x8<S>,
) {
    if cx_shift == 1 {
        let (ulo, _) = u8x16::from_slice(simd, &urow[cx..cx + 16]).widen();
        let (vlo, _) = u8x16::from_slice(simd, &vrow[cx..cx + 16]).widen();
        (
            ulo.zip_low(ulo),
            ulo.zip_high(ulo),
            vlo.zip_low(vlo),
            vlo.zip_high(vlo),
        )
    } else {
        let (ulo, uhi) = u8x16::from_slice(simd, &urow[cx..cx + 16]).widen();
        let (vlo, vhi) = u8x16::from_slice(simd, &vrow[cx..cx + 16]).widen();
        (ulo, uhi, vlo, vhi)
    }
}

#[inline(always)]
fn convert_group<S: Simd>(
    simd: S,
    y16: fearless_simd::u16x8<S>,
    u16v: fearless_simd::u16x8<S>,
    v16v: fearless_simd::u16x8<S>,
    out: &mut [u8],
) {
    let (r, g, b) = convert_channels(simd, y16, u16v, v16v);
    let px = r | (g << 8u32) | (b << 16u32) | ALPHA;
    // lane bytes in LE order are exactly the RGBA byte layout
    let bytes: fearless_simd::u8x32<S> = px.bitcast();
    bytes.store_slice(out);
}

#[inline(always)]
fn convert_group_with_keys<S: Simd>(
    simd: S,
    y16: fearless_simd::u16x8<S>,
    u16v: fearless_simd::u16x8<S>,
    v16v: fearless_simd::u16x8<S>,
    out: &mut [u8],
    keys: &mut [u32],
) {
    let (r, g, b) = convert_channels(simd, y16, u16v, v16v);
    let px = r | (g << 8u32) | (b << 16u32) | ALPHA;
    let bytes: fearless_simd::u8x32<S> = px.bitcast();
    bytes.store_slice(out);
    (((r >> 2u32) << 12u32) | ((g >> 2u32) << 6u32) | (b >> 2u32))
        .bitcast::<u32x8<S>>()
        .store_slice(keys);
}

#[inline(always)]
fn convert_group_to_rgb_keys<S: Simd>(
    simd: S,
    y16: fearless_simd::u16x8<S>,
    u16v: fearless_simd::u16x8<S>,
    v16v: fearless_simd::u16x8<S>,
    keys: &mut [u32],
) {
    let (r, g, b) = convert_channels(simd, y16, u16v, v16v);
    ((r << 16u32) | (g << 8u32) | b)
        .bitcast::<u32x8<S>>()
        .store_slice(keys);
}

#[inline(always)]
fn convert_channels<S: Simd>(
    simd: S,
    y16: fearless_simd::u16x8<S>,
    u16v: fearless_simd::u16x8<S>,
    v16v: fearless_simd::u16x8<S>,
) -> (i32x8<S>, i32x8<S>, i32x8<S>) {
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
    (r, g, b)
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

/// Activity attenuation only. Sources that emitted their grid keys while
/// constructing the RGBA row use this instead of rereading the row to
/// derive those keys a second time.
pub fn bn_activity(level: Level, cur: &[u8], prev: &[u8], gate: u32, att: &mut [u32]) {
    fearless_simd::dispatch!(level, simd => bn_activity_impl(simd, cur, prev, gate, att))
}

#[inline(always)]
fn bn_activity_impl<S: Simd>(simd: S, cur: &[u8], prev: &[u8], gate: u32, att: &mut [u32]) {
    let w = att.len();
    if w == 0 {
        return;
    }
    let g64 = gate as i32 + 64;
    att[0] = att_scalar(cur, prev, 0, g64);
    let zero = i32x16::splat(simd, 0);
    let full = i32x16::splat(simd, 256);
    let g64v = i32x16::splat(simd, g64);
    let mut i = 1usize;
    while i + 16 <= w {
        let px = px16(simd, &cur[i * 4..]);
        let pl = px16(simd, &cur[(i - 1) * 4..]);
        let pu = px16(simd, &prev[i * 4..]);
        let act = sad3(simd, px, pl) + sad3(simd, px, pu);
        let a = ((g64v - act) << 2u32).max(zero).min(full);
        a.bitcast::<u32x16<S>>().store_slice(&mut att[i..i + 16]);
        i += 16;
    }
    while i < w {
        att[i] = att_scalar(cur, prev, i, g64);
        i += 1;
    }
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

/// Summed absolute difference over all four bytes of two packed pixels
/// (alpha included): where alpha is constant this equals `sad3`; where it
/// changes the difference simply adds to the distance, which is the
/// intended effect for the prefilters (a pixel whose alpha moved is not
/// "the same" pixel) and saves the separate alpha compare.
#[inline(always)]
fn sad4<S: Simd>(simd: S, a: u32x16<S>, b: u32x16<S>) -> i32x16<S> {
    let ab = a.bitcast::<u8x64<S>>();
    let bb = b.bitcast::<u8x64<S>>();
    let d = (ab.max(bb) - ab.min(bb)).bitcast::<u32x16<S>>();
    let m = u32x16::splat(simd, 0x00FF_00FF);
    let pairs = (d & m) + ((d >> 8u32) & m); // r+g in low 16, b+a in high 16
    ((pairs & u32x16::splat(simd, 0xFFFF)) + (pairs >> 16u32)).bitcast()
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

// ---------------------------------------------------------------------------
// --hold with running mean, and --smooth (see input::hold / input::smooth)

use crate::input::hold::{MEAN_RATE, MEAN_SHIFT};

/// A per-pixel 32-bit mask widened to the 16-bit-lane layout of the
/// running mean (4 lanes per pixel): bytes of the mask are 0xFF/0, so
/// after widening each u16 lane is 0x00FF/0 and an equality test gives
/// the full-width lane mask.
#[inline(always)]
fn widen_mask<S: Simd>(simd: S, m: mask32x16<S>) -> (mask16x32<S>, mask16x32<S>) {
    let bytes: u8x64<S> = m
        .select(u32x16::splat(simd, u32::MAX), u32x16::splat(simd, 0))
        .bitcast();
    let (lo, hi): (u16x32<S>, u16x32<S>) = bytes.widen();
    let ff = u16x32::splat(simd, 0xFF);
    (lo.simd_eq(ff), hi.simd_eq(ff))
}

/// Mean update for 32 lanes: while held, m += (cur7 - m) >> 3; on reset
/// m = cur7 (cur7 = sample << 7).
#[inline(always)]
fn mean_step<S: Simd>(cur: u16x32<S>, m: i16x32<S>, keep: mask16x32<S>) -> i16x32<S> {
    let cur7: i16x32<S> = (cur << MEAN_SHIFT).bitcast();
    let tracked = m + ((cur7 - m) >> MEAN_RATE);
    keep.select(tracked, cur7)
}

/// Rounded 8-bit value of 64 lanes of 8.7 mean (two vectors).
#[inline(always)]
fn mean_round64<S: Simd>(simd: S, lo: i16x32<S>, hi: i16x32<S>) -> u8x64<S> {
    let half = i16x32::splat(simd, 1 << (MEAN_SHIFT - 1));
    let r_lo: u16x32<S> = ((lo + half) >> MEAN_SHIFT).bitcast();
    let r_hi: u16x32<S> = ((hi + half) >> MEAN_SHIFT).bitcast();
    r_lo.narrow(r_hi)
}

/// `--hold` with a running mean, packed RGBA (see `input::hold::rgba_mean`).
/// `mean` is one i16 per byte of `cur`. Result stored to `cur` and `prev`.
/// Two more things ride along in the same pass so the hold thread makes
/// one traversal per frame: the unmodified input is mirrored into
/// `raw_prev` (the next frame's noise reference) after its previous
/// contents have been compared against `cur` into `hist`, a histogram of
/// per-pixel L1 change (every fourth vector of 16 pixels — a quarter of
/// the frame is plenty for a quantile) that sizes the *next* frame's
/// window. Reading raw_prev before overwriting it is what makes the
/// fusion work: the reference is consumed and replaced in one step.
#[allow(clippy::too_many_arguments)]
pub fn hold_rgba_mean(
    level: Level,
    cur: &mut [u8],
    prev: &mut [u8],
    mean: &mut [i16],
    raw_prev: &mut [u8],
    hist: &mut [u32; 256],
    t: u32,
    tmax: u32,
) {
    fearless_simd::dispatch!(level, simd => hold_rgba_mean_impl(simd, cur, prev, mean, raw_prev, hist, t, tmax))
}

/// Histogram sampling: one vector in this many.
pub const HOLD_HIST_STRIDE: usize = 4;

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn hold_rgba_mean_impl<S: Simd>(
    simd: S,
    cur: &mut [u8],
    prev: &mut [u8],
    mean: &mut [i16],
    raw_prev: &mut [u8],
    hist: &mut [u32; 256],
    t: u32,
    tmax: u32,
) {
    let n = cur.len() / 4;
    let tv = i32x16::splat(simd, t as i32);
    let tmv = i32x16::splat(simd, tmax as i32);
    let cap = i32x16::splat(simd, 255);
    let mut i = 0usize;
    let mut v = 0usize;
    while i + 16 <= n {
        let a = px16(simd, &cur[i * 4..]);
        if v.is_multiple_of(HOLD_HIST_STRIDE) {
            let r = px16(simd, &raw_prev[i * 4..]);
            let d = sad4(simd, a, r).min(cap);
            let arr: [i32; 16] = d.into();
            for x in arr {
                hist[x as usize] += 1;
            }
        }
        a.bitcast::<u8x64<S>>()
            .store_slice(&mut raw_prev[i * 4..i * 4 + 64]);
        let b = px16(simd, &prev[i * 4..]);
        let m_lo = i16x32::from_slice(simd, &mean[i * 4..i * 4 + 32]);
        let m_hi = i16x32::from_slice(simd, &mean[i * 4 + 32..i * 4 + 64]);
        let m8: u32x16<S> = mean_round64(simd, m_lo, m_hi).bitcast();
        // alpha rides along in the L1 (the mean's alpha lane tracks the
        // input like any other, so it contributes 0 where alpha is static)
        let close = sad4(simd, a, m8).simd_lt(tv);
        let near = sad4(simd, a, b).simd_lt(tmv);
        let keep = close & near;
        let r = keep.select(b, a).bitcast::<u8x64<S>>();
        r.store_slice(&mut cur[i * 4..i * 4 + 64]);
        r.store_slice(&mut prev[i * 4..i * 4 + 64]);
        let (c_lo, c_hi) = a.bitcast::<u8x64<S>>().widen();
        let (k_lo, k_hi) = widen_mask(simd, keep);
        mean_step(c_lo, m_lo, k_lo).store_slice(&mut mean[i * 4..i * 4 + 32]);
        mean_step(c_hi, m_hi, k_hi).store_slice(&mut mean[i * 4 + 32..i * 4 + 64]);
        i += 16;
        v += 1;
    }
    let tail = i * 4;
    // tail pixels: sampled into the histogram only if the stride lands
    // on them, mirrored into raw_prev before the hold rewrites cur
    if v.is_multiple_of(HOLD_HIST_STRIDE) {
        for p in i..n {
            let d: u32 = (0..4)
                .map(|k| cur[p * 4 + k].abs_diff(raw_prev[p * 4 + k]) as u32)
                .sum();
            hist[d.min(255) as usize] += 1;
        }
    }
    raw_prev[tail..].copy_from_slice(&cur[tail..]);
    crate::input::hold::rgba_mean(&mut cur[tail..], &prev[tail..], &mut mean[tail..], t, tmax);
    prev[tail..].copy_from_slice(&cur[tail..]);
}

/// Rounded division of 16 lanes of non-negative sums by counts in
/// 1..=25: (acc + cnt/2) / cnt, computed in f32. The quotient's
/// fractional part is a multiple of 1/cnt >= 0.04, so a 0.01 nudge before
/// truncation makes the result exactly the integer division despite
/// float rounding.
#[inline(always)]
fn div_round16<S: Simd>(simd: S, acc: i32x16<S>, cnt: i32x16<S>, rcp: f32x16<S>) -> i32x16<S> {
    let q = f32x16::float_from(acc + (cnt >> 1u32)) * rcp + f32x16::splat(simd, 0.01);
    i32x16::truncate_from(q)
}

/// One output row of the range-limited 5x5 box filter, packed RGBA (see
/// `input::smooth::rgba_row`). `padded` is the PAD-replicated frame.
pub fn smooth_rgba_row(level: Level, padded: &[u8], w: usize, y: usize, s: u32, out: &mut [u8]) {
    fearless_simd::dispatch!(level, simd => smooth_rgba_row_impl(simd, padded, w, y, s, out))
}

#[inline(always)]
fn smooth_rgba_row_impl<S: Simd>(
    simd: S,
    padded: &[u8],
    w: usize,
    y: usize,
    s: u32,
    out: &mut [u8],
) {
    use crate::input::smooth::{PAD, WIN};
    let pw = w + 2 * PAD;
    let sv = i32x16::splat(simd, s as i32);
    let zero_u = u32x16::splat(simd, 0);
    let rb_mask = u32x16::splat(simd, 0x00FF_00FF);
    let ff = u32x16::splat(simd, 0xFF);
    let one_hi = u32x16::splat(simd, 1 << 16);
    let lo16 = u32x16::splat(simd, 0xFFFF);
    let alpha_mask = u32x16::splat(simd, 0xFF00_0000);
    let mut x = 0usize;
    while x + 16 <= w {
        let ci = ((y + PAD) * pw + x + PAD) * 4;
        let c = px16(simd, &padded[ci..]);
        // r and b sums share one accumulator (each <= 25*255 fits 16
        // bits); g shares another with the neighbour count in its high
        // half, so a passing neighbour costs two selects and two adds
        let mut acc_rb = zero_u;
        let mut acc_gc = zero_u;
        for dy in 0..WIN {
            let base = ((y + dy) * pw + x) * 4;
            for dx in 0..WIN {
                let nb = px16(simd, &padded[base + dx * 4..]);
                let ok = sad4(simd, nb, c).simd_lt(sv);
                acc_rb += ok.select(nb & rb_mask, zero_u);
                acc_gc += ok.select(((nb >> 8u32) & ff) | one_hi, zero_u);
            }
        }
        let cnt: i32x16<S> = (acc_gc >> 16u32).bitcast();
        let rcp = f32x16::splat(simd, 1.0) / f32x16::float_from(cnt);
        let r = div_round16(simd, (acc_rb & lo16).bitcast(), cnt, rcp);
        let b = div_round16(simd, (acc_rb >> 16u32).bitcast(), cnt, rcp);
        let g = div_round16(simd, (acc_gc & lo16).bitcast(), cnt, rcp);
        let px: u32x16<S> = r.bitcast::<u32x16<S>>()
            | (g.bitcast::<u32x16<S>>() << 8u32)
            | (b.bitcast::<u32x16<S>>() << 16u32)
            | (c & alpha_mask);
        px.bitcast::<u8x64<S>>()
            .store_slice(&mut out[x * 4..x * 4 + 64]);
        x += 16;
    }
    if x < w {
        // scalar tail over the padded frame's remaining columns
        let mut tmp = vec![0u8; (w - x) * 4];
        tail_rgba(padded, w, y, x, s, &mut tmp);
        out[x * 4..].copy_from_slice(&tmp);
    }
}

/// Scalar tail for `smooth_rgba_row_impl`: columns x0.. of row y.
fn tail_rgba(padded: &[u8], w: usize, y: usize, x0: usize, s: u32, out: &mut [u8]) {
    use crate::input::smooth::{PAD, WIN};
    let pw = w + 2 * PAD;
    for (k, o) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let x = x0 + k;
        let ci = ((y + PAD) * pw + x + PAD) * 4;
        let c = &padded[ci..ci + 4];
        let mut acc = [0u32; 3];
        let mut cnt = 0u32;
        for dy in 0..WIN {
            for dx in 0..WIN {
                let ni = ((y + dy) * pw + x + dx) * 4;
                let n = &padded[ni..ni + 4];
                let d = n[0].abs_diff(c[0]) as u32
                    + n[1].abs_diff(c[1]) as u32
                    + n[2].abs_diff(c[2]) as u32
                    + n[3].abs_diff(c[3]) as u32;
                if d < s {
                    acc[0] += n[0] as u32;
                    acc[1] += n[1] as u32;
                    acc[2] += n[2] as u32;
                    cnt += 1;
                }
            }
        }
        for ch in 0..3 {
            o[ch] = ((acc[ch] + cnt / 2) / cnt) as u8;
        }
        o[3] = c[3];
    }
}

#[cfg(test)]
mod hold_mean_smooth_tests {
    use super::*;
    use crate::input::{hold, smooth};

    fn noise(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 24) as u8
            })
            .collect()
    }
    fn jitter(base: &[u8], seed: u32, amp: i32) -> Vec<u8> {
        let j = noise(base.len(), seed);
        base.iter()
            .zip(&j)
            .map(|(&b, &r)| (b as i32 + (r as i32 % (2 * amp + 1)) - amp).clamp(0, 255) as u8)
            .collect()
    }

    #[test]
    fn hold_rgba_mean_matches_scalar() {
        let n = 100; // 6 vectors + 4-pixel tail
        let base = noise(n * 4, 1);
        let mut cur = jitter(&base, 2, 3);
        for i in (0..n * 4).step_by(37) {
            cur[i] = cur[i].wrapping_add(50);
        }
        cur[3] = 0; // one alpha change
        let mean: Vec<i16> = base
            .iter()
            .map(|&b| (b as i16) << hold::MEAN_SHIFT)
            .collect();
        let mut mean_j = mean.clone();
        for (k, m) in mean_j.iter_mut().enumerate() {
            *m += ((k % 7) as i16 - 3) * 40; // off-centre means
        }
        let (mut want, mut m_want) = (cur.clone(), mean_j.clone());
        hold::rgba_mean(&mut want, &base, &mut m_want, 8, 12);
        let (mut got, mut prev, mut m_got) = (cur.clone(), base.clone(), mean_j.clone());
        let mut raw_prev = base.clone();
        let mut hist = [0u32; 256];
        hold_rgba_mean(
            level(),
            &mut got,
            &mut prev,
            &mut m_got,
            &mut raw_prev,
            &mut hist,
            8,
            12,
        );
        assert_eq!(got, want);
        assert_eq!(prev, want);
        assert_eq!(m_got, m_want);
        assert_ne!(got, cur);
        assert_eq!(raw_prev, cur, "raw input mirrored for the next frame");
        // histogram: sampled vectors of |cur - base| (vector 0 and 4 of 6,
        // plus the 4-pixel tail at vector index 6)
        let mut want_h = [0u32; 256];
        for p in (0..n).filter(|&p| p / 16 % HOLD_HIST_STRIDE == 0) {
            let d: u32 = (0..4)
                .map(|k| cur[p * 4 + k].abs_diff(base[p * 4 + k]) as u32)
                .sum();
            want_h[d.min(255) as usize] += 1;
        }
        assert_eq!(hist, want_h);
    }

    #[test]
    fn smooth_rgba_row_matches_scalar() {
        let (w, h) = (83usize, 9usize); // 5 vectors + 3-pixel tail
        let base: Vec<u8> = (0..w * h * 4)
            .map(|i| {
                let (x, ch) = ((i / 4) % w, i % 4);
                if ch == 3 {
                    255
                } else if x < 40 {
                    90
                } else {
                    170
                }
            })
            .collect();
        let mut frame = jitter(&base, 5, 5);
        for i in (3..w * h * 4).step_by(4 * 53) {
            frame[i] = 0; // some alpha holes
        }
        let mut pad = Vec::new();
        smooth::pad(&frame, w, h, 4, &mut pad);
        for y in 0..h {
            let mut want = vec![0u8; w * 4];
            let mut got = vec![0u8; w * 4];
            smooth::rgba_row(&pad, w, y, 24, &mut want);
            smooth_rgba_row(level(), &pad, w, y, 24, &mut got);
            assert_eq!(got, want, "row {y}");
        }
    }
}

/// Histogram of per-pixel L1 change (over all four bytes, saturating at
/// 255) between two packed-RGBA frames, for the adaptive hold window.
pub fn delta_hist_rgba(level: Level, cur: &[u8], prev: &[u8], hist: &mut [u32; 256]) {
    fearless_simd::dispatch!(level, simd => delta_hist_rgba_impl(simd, cur, prev, hist))
}

#[inline(always)]
fn delta_hist_rgba_impl<S: Simd>(simd: S, cur: &[u8], prev: &[u8], hist: &mut [u32; 256]) {
    let n = cur.len() / 4;
    let cap = i32x16::splat(simd, 255);
    let mut i = 0usize;
    while i + 16 <= n {
        let d = sad4(simd, px16(simd, &cur[i * 4..]), px16(simd, &prev[i * 4..])).min(cap);
        let arr: [i32; 16] = d.into();
        for v in arr {
            hist[v as usize] += 1;
        }
        i += 16;
    }
    for p in i..n {
        let d: u32 = (0..4)
            .map(|k| cur[p * 4 + k].abs_diff(prev[p * 4 + k]) as u32)
            .sum();
        hist[d.min(255) as usize] += 1;
    }
}

#[cfg(test)]
mod delta_hist_tests {
    use super::*;

    #[test]
    fn delta_hist_matches_scalar() {
        let n = 101;
        let mut s = 5u32;
        let mut rnd = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 24) as u8
        };
        let cur: Vec<u8> = (0..n * 4).map(|_| rnd()).collect();
        let prev: Vec<u8> = (0..n * 4).map(|_| rnd()).collect();
        let mut want = [0u32; 256];
        for p in 0..n {
            let d: u32 = (0..4)
                .map(|k| cur[p * 4 + k].abs_diff(prev[p * 4 + k]) as u32)
                .sum();
            want[d.min(255) as usize] += 1;
        }
        let mut got = [0u32; 256];
        delta_hist_rgba(level(), &cur, &prev, &mut got);
        assert_eq!(got, want);
    }
}

// ---------------------------------------------------------------------------
// Dither::Auto banding score (see dither::BandGate)

/// Padding on each side of the padded index/flag rows used by
/// `band_score_row`: at least RUN, rounded to a vector-friendly 16.
pub const BAND_PAD: usize = 16;

/// Score one row of the nearest-index map. `idxp` is the row padded by
/// BAND_PAD bytes of 0xFF (never a palette index) on each side; `prev`
/// the previous row (unpadded) or None; `flat_prev` the previous row's
/// long-run flags (padded like `idxp`, zeros in the pads). Writes this
/// row's flags into `flat_cur` (same layout) and one candidate byte per
/// pixel into `cand`: bit 0 = horizontal contour candidate (x-1 | x), bit
/// 1 = vertical (prev | x). Candidates are boundaries whose two endpoints
/// both lie inside same-index runs of at least RUN pixels; the caller
/// applies the colour-pair test to the (sparse) candidates.
pub fn band_score_row(
    level: Level,
    idxp: &[u8],
    prev: Option<&[u8]>,
    flat_prev: &[u8],
    flat_cur: &mut [u8],
    ltmp: &mut [u8],
    cand: &mut [u8],
    w: usize,
) {
    fearless_simd::dispatch!(level, simd => band_score_row_impl(simd, idxp, prev, flat_prev, flat_cur, ltmp, cand, w))
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn band_score_row_impl<S: Simd>(
    simd: S,
    idxp: &[u8],
    prev: Option<&[u8]>,
    flat_prev: &[u8],
    flat_cur: &mut [u8],
    ltmp: &mut [u8],
    cand: &mut [u8],
    w: usize,
) {
    use crate::dither::BandGate;
    const P: usize = BAND_PAD;
    const RUN: usize = BandGate::RUN;
    let one = u8x64::splat(simd, 1);
    let zero = u8x64::splat(simd, 0);
    // L[x] = the RUN pixels ending at x are all equal (ltmp, padded layout,
    // zero beyond the row so the OR below reads nothing past the end)
    ltmp[..P].fill(0);
    ltmp[P + w..].fill(0);
    let mut x = 0usize;
    while x < w {
        let n = 64.min(w - x);
        let base = P + x;
        let c = load64(simd, idxp, base, n);
        let mut m = c.simd_eq(load64(simd, idxp, base - 1, n));
        for k in 2..RUN {
            m = m & c.simd_eq(load64(simd, idxp, base - k, n));
        }
        store64(m.select(one, zero), ltmp, base, n);
        x += 64;
    }
    // flat[x] = OR of L[x + s] for s in 0..RUN: x lies inside some run of
    // RUN equal pixels
    flat_cur[..P].fill(0);
    flat_cur[P + w..].fill(0);
    x = 0;
    while x < w {
        let n = 64.min(w - x);
        let base = P + x;
        let mut f = load64(simd, ltmp, base, n);
        for s in 1..RUN {
            f = f | load64(simd, ltmp, base + s, n);
        }
        store64(f, flat_cur, base, n);
        x += 64;
    }
    // candidates
    x = 0;
    while x < w {
        let n = 64.min(w - x);
        let base = P + x;
        let c = load64(simd, idxp, base, n);
        let f = load64(simd, flat_cur, base, n).simd_eq(one);
        let left = load64(simd, idxp, base - 1, n);
        let fl = load64(simd, flat_cur, base - 1, n).simd_eq(one);
        let h = f & fl & !c.simd_eq(left);
        let mut out = h.select(one, zero);
        if let Some(prev) = prev {
            let pv = load64(simd, prev, x, n);
            let fp = load64(simd, flat_prev, base, n).simd_eq(one);
            let v = f & fp & !c.simd_eq(pv);
            out = out | v.select(u8x64::splat(simd, 2), zero);
        }
        store64(out, cand, x, n);
        x += 64;
    }
}

/// 64 bytes from `buf[at..]`, zero-filled past `n` valid bytes (rows are
/// padded, so the load itself never runs past the buffer).
#[inline(always)]
fn load64<S: Simd>(simd: S, buf: &[u8], at: usize, n: usize) -> u8x64<S> {
    if n == 64 {
        u8x64::from_slice(simd, &buf[at..at + 64])
    } else {
        let mut tmp = [0u8; 64];
        let avail = (buf.len() - at).min(64);
        tmp[..avail].copy_from_slice(&buf[at..at + avail]);
        u8x64::from_slice(simd, &tmp)
    }
}

#[inline(always)]
fn store64<S: Simd>(v: u8x64<S>, buf: &mut [u8], at: usize, n: usize) {
    if n == 64 {
        v.store_slice(&mut buf[at..at + 64]);
    } else {
        let arr: [u8; 64] = v.into();
        buf[at..at + n].copy_from_slice(&arr[..n]);
    }
}

#[cfg(test)]
mod band_tests {
    use super::*;
    use crate::dither::BandGate;

    /// Scalar reference: flags and candidates as defined in the docs.
    fn reference(idx: &[u8], prev: Option<&[u8]>, flat_prev: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let w = idx.len();
        // flat[i] = the run containing i has length >= RUN
        let mut flat = vec![0u8; w];
        let mut start = 0;
        for i in 1..=w {
            if i == w || idx[i] != idx[start] {
                if i - start >= BandGate::RUN {
                    flat[start..i].fill(1);
                }
                start = i;
            }
        }
        let mut cand = vec![0u8; w];
        for i in 0..w {
            if i > 0 && idx[i] != idx[i - 1] && flat[i] == 1 && flat[i - 1] == 1 {
                cand[i] |= 1;
            }
            if let Some(p) = prev {
                if idx[i] != p[i] && flat[i] == 1 && flat_prev[i] == 1 {
                    cand[i] |= 2;
                }
            }
        }
        (flat, cand)
    }

    #[test]
    fn band_score_row_matches_reference() {
        let w = 203; // three full vectors + tail
        let mut s = 7u32;
        let mut rnd = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s >> 24
        };
        // runs of random length 1..20 of random indices
        let mut mk = |rnd: &mut dyn FnMut() -> u32| {
            let mut v = Vec::new();
            while v.len() < w {
                let len = (rnd() % 20 + 1) as usize;
                let c = (rnd() % 5) as u8;
                v.extend(std::iter::repeat_n(c, len));
            }
            v.truncate(w);
            v
        };
        let prev = mk(&mut rnd);
        let idx = mk(&mut rnd);
        let (flat_prev_ref, _) = reference(&prev, None, &[]);
        let (want_flat, want_cand) = reference(&idx, Some(&prev), &flat_prev_ref);
        const P: usize = BAND_PAD;
        let mut idxp = vec![0xFFu8; w + 2 * P];
        idxp[P..P + w].copy_from_slice(&idx);
        let mut flat_prev = vec![0u8; w + 2 * P];
        flat_prev[P..P + w].copy_from_slice(&flat_prev_ref);
        let mut flat_cur = vec![0u8; w + 2 * P];
        let mut ltmp = vec![0u8; w + 2 * P];
        let mut cand = vec![0u8; w];
        band_score_row(
            level(),
            &idxp,
            Some(&prev),
            &flat_prev,
            &mut flat_cur,
            &mut ltmp,
            &mut cand,
            w,
        );
        if flat_cur[P..P + w] != want_flat[..] {
            let i = (0..w).find(|&i| flat_cur[P + i] != want_flat[i]).unwrap();
            panic!(
                "flat mismatch at {i}: idx[{}..{}]={:?} got {:?} want {:?}",
                i.saturating_sub(9),
                (i + 9).min(w),
                &idx[i.saturating_sub(9)..(i + 9).min(w)],
                &flat_cur[P + i.saturating_sub(9)..P + (i + 9).min(w)],
                &want_flat[i.saturating_sub(9)..(i + 9).min(w)]
            );
        }
        if cand != want_cand {
            let i = (0..w).find(|&i| cand[i] != want_cand[i]).unwrap();
            panic!(
                "cand mismatch at {i}: got {} want {} idx {:?} prev {:?} flat {:?} flatp {:?}",
                cand[i],
                want_cand[i],
                &idx[i.saturating_sub(2)..(i + 2).min(w)],
                &prev[i.saturating_sub(2)..(i + 2).min(w)],
                &flat_cur[P + i.saturating_sub(2)..P + (i + 2).min(w)],
                &flat_prev[P + i.saturating_sub(2)..P + (i + 2).min(w)]
            );
        }
        assert!(
            want_cand.iter().any(|&c| c != 0),
            "test input must produce candidates"
        );
    }
}
