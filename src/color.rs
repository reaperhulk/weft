//! YUV (BT.601 limited range) -> RGBA conversion.

use crate::input::{Chroma, Frame};

// BT.601 limited-range coefficients in 16.16 fixed point (matching
// swscale's precision: 1.164, 1.596, -0.392, -0.813, 2.017).
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

/// Convert one stored frame to RGBA into `out` (len == w*h*4).
pub fn frame_to_rgba(frame: &Frame, w: usize, h: usize, chroma: Option<Chroma>, out: &mut [u8]) {
    match frame {
        Frame::Rgba(buf) => out.copy_from_slice(buf),
        Frame::Yuv(buf) => yuv_to_rgba(buf, w, h, chroma.unwrap(), out),
    }
}

fn yuv_to_rgba(buf: &[u8], w: usize, h: usize, chroma: Chroma, out: &mut [u8]) {
    let cw = (w + 1) / 2;
    let ch = (h + 1) / 2;
    let (y_plane, u_plane, v_plane, cx_shift, cy_shift, crow_w) = match chroma {
        Chroma::C420 => {
            let (y, rest) = buf.split_at(w * h);
            let (u, v) = rest.split_at(cw * ch);
            (y, u, v, 1u32, 1u32, cw)
        }
        Chroma::C422 => {
            let (y, rest) = buf.split_at(w * h);
            let (u, v) = rest.split_at(cw * h);
            (y, u, v, 1, 0, cw)
        }
        Chroma::C444 => {
            let (y, rest) = buf.split_at(w * h);
            let (u, v) = rest.split_at(w * h);
            (y, u, v, 0, 0, w)
        }
        Chroma::Mono => {
            let y = &buf[..w * h];
            for (dst, &yy) in out.chunks_exact_mut(4).zip(y.iter()) {
                let c = CY * (yy as i32 - 16);
                let g = clamp8((c + ROUND) >> 16);
                dst[0] = g;
                dst[1] = g;
                dst[2] = g;
                dst[3] = 255;
            }
            return;
        }
    };

    for y in 0..h {
        let yrow = &y_plane[y * w..y * w + w];
        let crow = (y >> cy_shift) * crow_w;
        let urow = &u_plane[crow..crow + crow_w];
        let vrow = &v_plane[crow..crow + crow_w];
        let orow = &mut out[y * w * 4..(y + 1) * w * 4];
        for (x, dst) in orow.chunks_exact_mut(4).enumerate() {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primaries_roundtrip_420() {
        // white / black / red-ish in BT.601 limited
        let w = 2;
        let h = 2;
        // Y plane: all white (235); U=V=128
        let buf = vec![235, 235, 235, 235, 128, 128];
        let mut out = vec![0u8; 16];
        yuv_to_rgba(&buf, w, h, Chroma::C420, &mut out);
        assert_eq!(&out[..4], &[255, 255, 255, 255]);

        let buf = vec![16, 16, 16, 16, 128, 128];
        yuv_to_rgba(&buf, w, h, Chroma::C420, &mut out);
        assert_eq!(&out[..4], &[0, 0, 0, 255]);
    }
}
