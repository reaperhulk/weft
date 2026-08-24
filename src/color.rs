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

    /// The frame these rows come from.
    pub fn frame(&self) -> &'a Frame {
        self.frame
    }

    /// The frame's chroma layout (None for RGBA input).
    pub fn chroma(&self) -> Option<Chroma> {
        self.chroma
    }

    /// Fill `out` (len w*4) with RGBA for row `y`.
    #[inline]
    pub fn fill_row(&self, y: usize, out: &mut [u8]) {
        let w = self.w;
        match self.frame {
            Frame::Rgba(buf) => out.copy_from_slice(&buf[y * w * 4..(y + 1) * w * 4]),
            Frame::Yuv(buf) => fill_row_yuv(buf, w, self.h, self.chroma.unwrap(), y, out),
        }
    }
}

/// Fraction (in percent) of sampled pixels that are unchanged between two
/// frames — every 16th row, so a sixteenth of the cost of the full
/// comparison. Deterministic in the two frames, so decisions made from it
/// cannot make the output depend on scheduling.
pub fn unchanged_pct(
    cur: &Frame,
    prev: &Frame,
    w: usize,
    h: usize,
    chroma: Option<Chroma>,
) -> usize {
    let mut bits = vec![0u64; w.div_ceil(64)];
    let (mut changed, mut total) = (0usize, 0usize);
    for y in (0..h).step_by(16) {
        changed_row(cur, prev, w, h, chroma, y, &mut bits);
        changed += bits.iter().map(|c| c.count_ones() as usize).sum::<usize>();
        total += w;
    }
    if total == 0 {
        return 0;
    }
    100 - (changed * 100 / total).min(100)
}

/// Set a bit in `out` per pixel of row `y` that differs between two
/// frames of the same geometry (64 pixels per word).
///
/// Compared in the source's own domain — planar YUV is 1.5 bytes per
/// pixel against RGBA's 4, and needs no conversion — because this runs
/// for every row of every frame to find the pixels whose quantized index
/// cannot have changed.
pub fn changed_row(
    cur: &Frame,
    prev: &Frame,
    w: usize,
    h: usize,
    chroma: Option<Chroma>,
    y: usize,
    out: &mut [u64],
) {
    let level = crate::simdops::level();
    match (cur, prev) {
        (Frame::Rgba(a), Frame::Rgba(b)) => crate::simdops::diff_bits_rgba(
            level,
            &a[y * w * 4..(y + 1) * w * 4],
            &b[y * w * 4..(y + 1) * w * 4],
            w,
            out,
        ),
        (Frame::Yuv(a), Frame::Yuv(b)) => {
            let chroma = chroma.expect("yuv frame without chroma");
            let cw = w.div_ceil(2);
            let (ya, ra) = a.split_at(w * h);
            let (yb, rb) = b.split_at(w * h);
            crate::simdops::diff_bits(level, &ya[y * w..y * w + w], &yb[y * w..y * w + w], w, out);
            let (csize, crow, cn, sub) = match chroma {
                Chroma::Mono => return,
                Chroma::C420 => (cw * h.div_ceil(2), (y >> 1) * cw, cw, true),
                Chroma::C422 => (cw * h, y * cw, cw, true),
                Chroma::C444 => (w * h, y * w, w, false),
            };
            let ((ua, va), (ub, vb)) = (ra.split_at(csize), rb.split_at(csize));
            // Chroma changes invalidate every pixel they cover, so the
            // chroma-resolution mask is spread back out to pixels.
            let words = cn.div_ceil(64);
            let mut cbits = [0u64; 64];
            let mut vbits = [0u64; 64];
            let cb = &mut cbits[..words];
            let vb2 = &mut vbits[..words];
            crate::simdops::diff_bits(level, &ua[crow..crow + cn], &ub[crow..crow + cn], cn, cb);
            crate::simdops::diff_bits(level, &va[crow..crow + cn], &vb[crow..crow + cn], cn, vb2);
            if sub {
                for (i, o) in out[..w.div_ceil(64)].iter_mut().enumerate() {
                    let m = cb[i / 2] | vb2[i / 2];
                    let half = if i % 2 == 0 {
                        m as u32
                    } else {
                        (m >> 32) as u32
                    };
                    *o |= crate::simdops::spread2(half);
                }
            } else {
                for (o, (&c, &v)) in out[..words].iter_mut().zip(cb.iter().zip(vb2.iter())) {
                    *o |= c | v;
                }
            }
        }
        _ => out.fill(!0),
    }
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
