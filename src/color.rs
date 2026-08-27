//! YUV (BT.601 limited range) -> RGBA conversion.
//!
//! Conversion is row-based so hot loops can pull one RGBA row at a time
//! into a small scratch buffer (L1-resident) instead of materializing whole
//! RGBA frames.

use crate::input::{Chroma, Frame};

// BT.601 limited-range coefficients in 16.16 fixed point with round-to-
// nearest (1.164, 1.596, -0.392, -0.813, 2.017). Note: swscale truncates
// where this rounds, so outputs can differ from ffmpeg's by ±1-2 — this is
// the mathematically closer result.
const CY: i32 = 76309;
const CRV: i32 = 104597;
const CGU: i32 = -25675;
const CGV: i32 = -53279;
const CBU: i32 = 132201;
const ROUND: i32 = 1 << 15;

#[inline(always)]
fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Per-frame source of RGBA rows.
pub struct RowSource<'a> {
    frame: &'a Frame,
    w: usize,
    h: usize,
    chroma: Option<Chroma>,
}

impl<'a> RowSource<'a> {
    pub fn new(frame: &'a Frame, w: usize, h: usize, chroma: Option<Chroma>) -> Self {
        RowSource {
            frame,
            w,
            h,
            chroma,
        }
    }

    /// Borrow row `y` as RGBA when the frame is stored that way, so hot
    /// callers can read it in place instead of copying it into scratch.
    /// Returns None for the formats that need conversion (`Rgb`, `Yuv`);
    /// those callers fall back to `fill_row`.
    #[inline]
    pub fn rgba_row(&self, y: usize) -> Option<&'a [u8]> {
        match self.frame {
            Frame::Rgba(buf) => Some(&buf[y * self.w * 4..(y + 1) * self.w * 4]),
            Frame::Rgb(_) | Frame::Yuv(_) => None,
        }
    }

    /// Fill `out` (len w*4) with RGBA for row `y`.
    #[inline]
    pub fn fill_row(&self, y: usize, out: &mut [u8]) {
        let w = self.w;
        match self.frame {
            Frame::Rgba(buf) => out.copy_from_slice(&buf[y * w * 4..(y + 1) * w * 4]),
            Frame::Rgb(buf) => expand_row_rgb(&buf[y * w * 3..(y + 1) * w * 3], out),
            Frame::Yuv(buf) => fill_row_yuv(buf, w, self.h, self.chroma.unwrap(), y, out),
        }
    }

    /// Fill an RGBA row and, when the source is intrinsically opaque,
    /// emit its blue-noise grid keys as part of the same conversion pass.
    /// Returns whether `keys` was filled.
    #[inline]
    pub fn fill_row_with_grid_keys(&self, y: usize, out: &mut [u8], keys: &mut [u32]) -> bool {
        let w = self.w;
        match self.frame {
            Frame::Rgba(buf) => {
                out.copy_from_slice(&buf[y * w * 4..(y + 1) * w * 4]);
                false
            }
            Frame::Rgb(buf) => {
                expand_row_rgb_with_keys(&buf[y * w * 3..(y + 1) * w * 3], out, keys);
                true
            }
            Frame::Yuv(buf) => {
                fill_row_yuv_with_keys(buf, w, self.h, self.chroma.unwrap(), y, out, keys);
                true
            }
        }
    }

    /// Whether histogram rows can be converted straight to canonical
    /// `0xRRGGBB` keys without first materializing RGBA.
    #[inline]
    pub fn has_direct_rgb_keys(&self) -> bool {
        matches!(self.frame, Frame::Yuv(_))
    }

    /// Whether the frame representation guarantees opaque pixels. Pass 1
    /// packs opaque RGBA frames to RGB, while every supported YUV format is
    /// intrinsically opaque.
    #[inline]
    pub fn is_intrinsically_opaque(&self) -> bool {
        matches!(self.frame, Frame::Rgb(_) | Frame::Yuv(_))
    }

    /// Fill canonical `0xRRGGBB` histogram keys. Call only when
    /// `has_direct_rgb_keys` is true.
    #[inline]
    pub fn fill_rgb_keys(&self, y: usize, keys: &mut [u32]) {
        match self.frame {
            Frame::Yuv(buf) => {
                fill_row_yuv_rgb_keys(buf, self.w, self.h, self.chroma.unwrap(), y, keys)
            }
            Frame::Rgba(_) | Frame::Rgb(_) => unreachable!("source has no direct RGB-key path"),
        }
    }
}

/// Re-expand a packed RGB row to RGBA. Frames are only stored as RGB when
/// pass 1 found every pixel opaque, so the synthesized alpha is 255.
#[inline]
fn expand_row_rgb(row: &[u8], out: &mut [u8]) {
    for (dst, src) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(row.as_chunks::<3>().0)
    {
        *dst = [src[0], src[1], src[2], 255];
    }
}

#[inline]
fn expand_row_rgb_with_keys(row: &[u8], out: &mut [u8], keys: &mut [u32]) {
    for ((dst, key), src) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(keys)
        .zip(row.as_chunks::<3>().0)
    {
        *dst = [src[0], src[1], src[2], 255];
        *key = grid_key(src[0], src[1], src[2]);
    }
}

#[inline(always)]
fn grid_key(r: u8, g: u8, b: u8) -> u32 {
    (((r as u32) >> 2) << 12) | (((g as u32) >> 2) << 6) | ((b as u32) >> 2)
}

/// Pack an all-opaque RGBA frame down to RGB, dropping the alpha byte.
/// Callers must have established that no pixel is transparent (alpha
/// < 128) — see `Frame::Rgb`.
pub fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    let px = rgba.as_chunks::<4>().0;
    let mut rgb = vec![0u8; px.len() * 3];
    for (dst, src) in rgb.as_chunks_mut::<3>().0.iter_mut().zip(px) {
        *dst = [src[0], src[1], src[2]];
    }
    rgb
}

#[inline]
fn fill_row_yuv(buf: &[u8], w: usize, h: usize, chroma: Chroma, y: usize, out: &mut [u8]) {
    let cw = w.div_ceil(2);
    match chroma {
        Chroma::Mono => {
            let yrow = &buf[y * w..y * w + w];
            for (dst, &yy) in out.as_chunks_mut::<4>().0.iter_mut().zip(yrow.iter()) {
                let g = clamp8((CY * (yy as i32 - 16) + ROUND) >> 16);
                dst[0] = g;
                dst[1] = g;
                dst[2] = g;
                dst[3] = 255;
            }
        }
        Chroma::C420 => {
            let ch = h.div_ceil(2);
            let (yp, rest) = buf.split_at(w * h);
            let (up, vp) = rest.split_at(cw * ch);
            let crow = (y >> 1) * cw;
            convert_row(
                &yp[y * w..y * w + w],
                &up[crow..crow + cw],
                &vp[crow..crow + cw],
                1,
                out,
            );
        }
        Chroma::C422 => {
            let (yp, rest) = buf.split_at(w * h);
            let (up, vp) = rest.split_at(cw * h);
            let crow = y * cw;
            convert_row(
                &yp[y * w..y * w + w],
                &up[crow..crow + cw],
                &vp[crow..crow + cw],
                1,
                out,
            );
        }
        Chroma::C444 => {
            let (yp, rest) = buf.split_at(w * h);
            let (up, vp) = rest.split_at(w * h);
            let crow = y * w;
            convert_row(
                &yp[y * w..y * w + w],
                &up[crow..crow + w],
                &vp[crow..crow + w],
                0,
                out,
            );
        }
    }
}

#[inline]
fn fill_row_yuv_with_keys(
    buf: &[u8],
    w: usize,
    h: usize,
    chroma: Chroma,
    y: usize,
    out: &mut [u8],
    keys: &mut [u32],
) {
    let cw = w.div_ceil(2);
    match chroma {
        Chroma::Mono => {
            let yrow = &buf[y * w..y * w + w];
            for ((dst, key), &yy) in out.as_chunks_mut::<4>().0.iter_mut().zip(keys).zip(yrow) {
                let g = clamp8((CY * (yy as i32 - 16) + ROUND) >> 16);
                *dst = [g, g, g, 255];
                *key = grid_key(g, g, g);
            }
        }
        Chroma::C420 => {
            let ch = h.div_ceil(2);
            let (yp, rest) = buf.split_at(w * h);
            let (up, vp) = rest.split_at(cw * ch);
            let crow = (y >> 1) * cw;
            convert_row_with_keys(
                &yp[y * w..y * w + w],
                &up[crow..crow + cw],
                &vp[crow..crow + cw],
                1,
                out,
                keys,
            );
        }
        Chroma::C422 => {
            let (yp, rest) = buf.split_at(w * h);
            let (up, vp) = rest.split_at(cw * h);
            let crow = y * cw;
            convert_row_with_keys(
                &yp[y * w..y * w + w],
                &up[crow..crow + cw],
                &vp[crow..crow + cw],
                1,
                out,
                keys,
            );
        }
        Chroma::C444 => {
            let (yp, rest) = buf.split_at(w * h);
            let (up, vp) = rest.split_at(w * h);
            let crow = y * w;
            convert_row_with_keys(
                &yp[y * w..y * w + w],
                &up[crow..crow + w],
                &vp[crow..crow + w],
                0,
                out,
                keys,
            );
        }
    }
}

#[inline]
fn fill_row_yuv_rgb_keys(
    buf: &[u8],
    w: usize,
    h: usize,
    chroma: Chroma,
    y: usize,
    keys: &mut [u32],
) {
    let cw = w.div_ceil(2);
    match chroma {
        Chroma::Mono => {
            for (key, &yy) in keys.iter_mut().zip(&buf[y * w..y * w + w]) {
                let g = clamp8((CY * (yy as i32 - 16) + ROUND) >> 16);
                *key = ((g as u32) << 16) | ((g as u32) << 8) | g as u32;
            }
        }
        Chroma::C420 => {
            let ch = h.div_ceil(2);
            let (yp, rest) = buf.split_at(w * h);
            let (up, vp) = rest.split_at(cw * ch);
            let crow = (y >> 1) * cw;
            convert_row_to_rgb_keys(
                &yp[y * w..y * w + w],
                &up[crow..crow + cw],
                &vp[crow..crow + cw],
                1,
                keys,
            );
        }
        Chroma::C422 => {
            let (yp, rest) = buf.split_at(w * h);
            let (up, vp) = rest.split_at(cw * h);
            let crow = y * cw;
            convert_row_to_rgb_keys(
                &yp[y * w..y * w + w],
                &up[crow..crow + cw],
                &vp[crow..crow + cw],
                1,
                keys,
            );
        }
        Chroma::C444 => {
            let (yp, rest) = buf.split_at(w * h);
            let (up, vp) = rest.split_at(w * h);
            let crow = y * w;
            convert_row_to_rgb_keys(
                &yp[y * w..y * w + w],
                &up[crow..crow + w],
                &vp[crow..crow + w],
                0,
                keys,
            );
        }
    }
}

#[inline(always)]
fn convert_row(yrow: &[u8], urow: &[u8], vrow: &[u8], cx_shift: u32, out: &mut [u8]) {
    // SIMD main blocks (byte-identical math), scalar tail.
    let done =
        crate::simdops::convert_row(crate::simdops::level(), yrow, urow, vrow, cx_shift, out);
    for (x, dst) in out.as_chunks_mut::<4>().0.iter_mut().enumerate().skip(done) {
        let cx = x >> cx_shift;
        let c = CY * (yrow[x] as i32 - 16);
        let d = urow[cx] as i32 - 128;
        let e = vrow[cx] as i32 - 128;
        dst[0] = clamp8((c + CRV * e + ROUND) >> 16);
        dst[1] = clamp8((c + CGU * d + CGV * e + ROUND) >> 16);
        dst[2] = clamp8((c + CBU * d + ROUND) >> 16);
        dst[3] = 255;
    }
}

#[inline(always)]
fn convert_row_with_keys(
    yrow: &[u8],
    urow: &[u8],
    vrow: &[u8],
    cx_shift: u32,
    out: &mut [u8],
    keys: &mut [u32],
) {
    let done = crate::simdops::convert_row_with_keys(
        crate::simdops::level(),
        yrow,
        urow,
        vrow,
        cx_shift,
        out,
        keys,
    );
    for x in done..yrow.len() {
        let cx = x >> cx_shift;
        let c = CY * (yrow[x] as i32 - 16);
        let d = urow[cx] as i32 - 128;
        let e = vrow[cx] as i32 - 128;
        let r = clamp8((c + CRV * e + ROUND) >> 16);
        let g = clamp8((c + CGU * d + CGV * e + ROUND) >> 16);
        let b = clamp8((c + CBU * d + ROUND) >> 16);
        out[x * 4..x * 4 + 4].copy_from_slice(&[r, g, b, 255]);
        keys[x] = grid_key(r, g, b);
    }
}

#[inline(always)]
fn convert_row_to_rgb_keys(yrow: &[u8], urow: &[u8], vrow: &[u8], cx_shift: u32, keys: &mut [u32]) {
    let done = crate::simdops::convert_row_to_rgb_keys(
        crate::simdops::level(),
        yrow,
        urow,
        vrow,
        cx_shift,
        keys,
    );
    for x in done..yrow.len() {
        let cx = x >> cx_shift;
        let c = CY * (yrow[x] as i32 - 16);
        let d = urow[cx] as i32 - 128;
        let e = vrow[cx] as i32 - 128;
        let r = clamp8((c + CRV * e + ROUND) >> 16);
        let g = clamp8((c + CGU * d + CGV * e + ROUND) >> 16);
        let b = clamp8((c + CBU * d + ROUND) >> 16);
        keys[x] = ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
    }
}

/// Convert a whole stored frame to RGBA (tests and non-hot paths).
#[cfg_attr(not(test), allow(dead_code))]
pub fn frame_to_rgba(frame: &Frame, w: usize, h: usize, chroma: Option<Chroma>, out: &mut [u8]) {
    let src = RowSource::new(frame, w, h, chroma);
    for y in 0..h {
        src.fill_row(y, &mut out[y * w * 4..(y + 1) * w * 4]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primaries_roundtrip_420() {
        let w = 2;
        let h = 2;
        let buf = vec![235, 235, 235, 235, 128, 128];
        let mut out = vec![0u8; 16];
        frame_to_rgba(&Frame::Yuv(buf), w, h, Some(Chroma::C420), &mut out);
        assert_eq!(&out[..4], &[255, 255, 255, 255]);

        let buf = vec![16, 16, 16, 16, 128, 128];
        frame_to_rgba(&Frame::Yuv(buf), w, h, Some(Chroma::C420), &mut out);
        assert_eq!(&out[..4], &[0, 0, 0, 255]);
    }

    #[test]
    fn rgb_frames_round_trip_with_opaque_alpha() {
        // 2x2 RGBA, all opaque -> packed RGB -> rows re-expanded to RGBA
        let w = 2;
        let h = 2;
        let rgba: Vec<u8> = (0..h as u8)
            .flat_map(|y| (0..w as u8).flat_map(move |x| [x * 10, y * 20, 30 + x, 255]))
            .collect();
        let rgb = rgba_to_rgb(&rgba);
        assert_eq!(rgb, vec![0, 0, 30, 10, 0, 31, 0, 20, 30, 10, 20, 31]);
        let frame = Frame::Rgb(rgb);
        let mut out = vec![0u8; w * h * 4];
        frame_to_rgba(&frame, w, h, None, &mut out);
        assert_eq!(out, rgba);
        // packed rows can't be borrowed as RGBA
        assert!(RowSource::new(&frame, w, h, None).rgba_row(0).is_none());
    }

    #[test]
    fn rgba_rows_are_borrowed_from_input() {
        let frame = Frame::Rgba((0..32).collect());
        let src = RowSource::new(&frame, 4, 2, None);
        assert_eq!(src.rgba_row(1).unwrap(), &(16..32).collect::<Vec<_>>());
        assert!(
            RowSource::new(&Frame::Yuv(vec![0; 6]), 2, 2, Some(Chroma::C420))
                .rgba_row(0)
                .is_none()
        );
    }

    #[test]
    fn fused_yuv_keys_match_rgba_conversion() {
        for chroma in [Chroma::Mono, Chroma::C420, Chroma::C422, Chroma::C444] {
            let w = 33usize;
            let h = 3usize;
            let cw = w.div_ceil(2);
            let len = match chroma {
                Chroma::Mono => w * h,
                Chroma::C420 => w * h + 2 * cw * h.div_ceil(2),
                Chroma::C422 => w * h + 2 * cw * h,
                Chroma::C444 => 3 * w * h,
            };
            let buf: Vec<u8> = (0..len)
                .map(|i| ((i * 73 + i / 7 * 29 + 17) & 0xff) as u8)
                .collect();
            let frame = Frame::Yuv(buf);
            let src = RowSource::new(&frame, w, h, Some(chroma));
            for y in 0..h {
                let mut plain = vec![0; w * 4];
                let mut fused = vec![0; w * 4];
                let mut grid = vec![0; w];
                let mut rgb = vec![0; w];
                src.fill_row(y, &mut plain);
                assert!(src.fill_row_with_grid_keys(y, &mut fused, &mut grid));
                src.fill_rgb_keys(y, &mut rgb);
                assert_eq!(fused, plain);
                for (i, px) in plain.as_chunks::<4>().0.iter().enumerate() {
                    assert_eq!(grid[i], grid_key(px[0], px[1], px[2]));
                    assert_eq!(
                        rgb[i],
                        ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | px[2] as u32
                    );
                }
            }
        }
    }

    #[test]
    fn c444_and_c422_rows() {
        // 2x2 444: distinct chroma per pixel
        let w = 2;
        let h = 2;
        let buf = vec![
            81, 145, 41, 210, // Y
            90, 54, 240, 16, // U
            240, 34, 110, 146, // V
        ];
        let mut out = vec![0u8; 16];
        frame_to_rgba(&Frame::Yuv(buf), w, h, Some(Chroma::C444), &mut out);
        // red-ish first pixel (bt601 red: Y81 U90 V240)
        assert!(
            out[0] > 200 && out[1] < 60 && out[2] < 60,
            "{:?}",
            &out[..4]
        );
    }
}
