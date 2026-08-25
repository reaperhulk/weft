//! stdin readers: yuv4mpegpipe (auto-detected) and raw RGBA.

use std::io::{self, BufRead, Read};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chroma {
    C420,
    C422,
    C444,
    Mono,
}

impl Chroma {
    pub fn frame_bytes(self, w: usize, h: usize) -> usize {
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        match self {
            Chroma::C420 => w * h + 2 * cw * ch,
            Chroma::C422 => w * h + 2 * cw * h,
            Chroma::C444 => 3 * w * h,
            Chroma::Mono => w * h,
        }
    }
}

/// A stored input frame: packed pixels or raw Y4M planes. Frames are kept
/// in the smallest form that reproduces their pixels exactly and converted
/// to RGBA rows on the fly in each parallel pass, trading a cheap
/// reconversion for a much smaller resident set — the whole clip stays
/// resident between passes, so a byte per pixel is a byte per pixel per
/// frame.
pub enum Frame {
    /// Raw RGBA, kept only for frames that actually carry transparency.
    Rgba(Vec<u8>),
    /// Packed RGB: an RGBA frame whose every pixel pass 1 found opaque, so
    /// the alpha plane is a constant and costs nothing to re-synthesize.
    /// A quarter smaller than `Rgba`, which is a quarter off the resident
    /// set for the overwhelmingly common alpha-free RGBA clip.
    Rgb(Vec<u8>),
    Yuv(Vec<u8>),
}

pub struct VideoIn {
    pub width: usize,
    pub height: usize,
    pub fps_num: u32,
    pub fps_den: u32,
    pub chroma: Option<Chroma>, // None => RGBA input
}

pub fn parse_y4m_header(line: &str) -> io::Result<VideoIn> {
    let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, format!("y4m: {m}"));
    let mut w = 0usize;
    let mut h = 0usize;
    let mut num = 25u32;
    let mut den = 1u32;
    let mut chroma = Chroma::C420;
    for tok in line.split_ascii_whitespace().skip(1) {
        let (tag, val) = tok.split_at(1);
        match tag {
            "W" => w = val.parse().map_err(|_| bad("bad W"))?,
            "H" => h = val.parse().map_err(|_| bad("bad H"))?,
            "F" => {
                let (n, d) = val.split_once(':').ok_or_else(|| bad("bad F"))?;
                num = n.parse().map_err(|_| bad("bad F num"))?;
                den = d.parse().map_err(|_| bad("bad F den"))?;
            }
            "C" => {
                chroma = if val.starts_with("420") {
                    Chroma::C420
                } else if val.starts_with("422") {
                    Chroma::C422
                } else if val.starts_with("444") && !val.contains("alpha") {
                    Chroma::C444
                } else if val.starts_with("mono") {
                    Chroma::Mono
                } else {
                    return Err(bad(&format!("unsupported colorspace C{val}")));
                };
                if val.contains("p10") || val.contains("p12") || val.contains("p16") {
                    return Err(bad(&format!("unsupported bit depth C{val}")));
                }
            }
            _ => {} // interlace, aspect, extensions: ignored
        }
    }
    if w == 0 || h == 0 {
        return Err(bad("missing W/H"));
    }
    if num == 0 || den == 0 {
        return Err(bad("bad frame rate"));
    }
    Ok(VideoIn {
        width: w,
        height: h,
        fps_num: num,
        fps_den: den,
        chroma: Some(chroma),
    })
}

/// Fill the caller's frame buffer (`buf.len()` bytes, already faulted in
/// — see the reader's prefault thread in main.rs) from the stream.
/// Returns None on immediate clean EOF; errors on a short read.
fn read_frame_into(r: &mut impl Read, mut buf: Vec<u8>, what: &str) -> io::Result<Option<Vec<u8>>> {
    let n = buf.len();
    let mut got = 0usize;
    while got < n {
        match r.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(k) => got += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    if got == 0 {
        return Ok(None);
    }
    if got < n {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("{what}: truncated frame ({got} of {n} bytes)"),
        ));
    }
    Ok(Some(buf))
}

/// Read the next Y4M frame (FRAME marker + planes) into `buf`, which must
/// hold exactly one frame (`Chroma::frame_bytes`), or None at EOF.
pub fn read_y4m_frame(r: &mut impl BufRead, buf: Vec<u8>) -> io::Result<Option<Frame>> {
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None); // clean EOF
    }
    if !line.starts_with("FRAME") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "y4m: expected FRAME marker",
        ));
    }
    match read_frame_into(r, buf, "y4m")? {
        Some(buf) => Ok(Some(Frame::Yuv(buf))),
        None => Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "y4m: FRAME marker with no payload",
        )),
    }
}

/// Read the next raw RGBA frame into `buf` (exactly w*h*4 bytes), or None
/// at EOF.
pub fn read_rgba_frame(r: &mut impl Read, buf: Vec<u8>) -> io::Result<Option<Frame>> {
    Ok(read_frame_into(r, buf, "raw rgba")?.map(Frame::Rgba))
}

/// Temporal hold: the per-pixel "keep the previous value" prefilter for
/// `--hold`. Compressed video carries a few LSB of noise on every pixel
/// of every frame, and on flat content that noise — not the picture —
/// decides which of two neighboring palette entries a pixel lands on.
/// Re-rolled every frame, it turns static fills into per-frame index
/// churn that defeats the delta encoder (measured on a flat cel-animation
/// clip: 64% of source pixels change frame-to-frame but only 22% by more
/// than 6/765, and `--dither none` still re-encoded 27% of pixels per
/// frame). Holding a pixel at its previous value while it moves by less
/// than a threshold makes static regions byte-identical across frames, so
/// they drop out of the delta entirely. A held pixel is never more than
/// the threshold away from its true value, and any larger change passes
/// through untouched, so motion and cuts are unaffected. Slow fades step
/// per pixel by up to the threshold instead of moving smoothly — keep it
/// small (about 8–12 for RGB L1).
///
/// `cur` is rewritten in place and then *is* the reference for the next
/// frame: a pixel that holds stays at the reference value, a pixel that
/// moves becomes the new reference, so drift can never accumulate beyond
/// one threshold. These are the scalar reference kernels; the reader
/// uses the SIMD versions in `simdops` (which also mirror the result into
/// the reference buffer) and falls back to these for vector tails.
pub mod hold {
    /// RGBA frames: hold a pixel when the L1 distance over RGB to the
    /// reference is below `t` and alpha is unchanged.
    pub fn rgba(cur: &mut [u8], prev: &[u8], t: u32) {
        debug_assert_eq!(cur.len(), prev.len());
        for (c, p) in cur
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(prev.as_chunks::<4>().0)
        {
            let d = c[0].abs_diff(p[0]) as u32
                + c[1].abs_diff(p[1]) as u32
                + c[2].abs_diff(p[2]) as u32;
            if d < t && c[3] == p[3] {
                *c = *p;
            }
        }
    }

    /// Planar (Y4M) frames: hold each sample independently when it moves
    /// by less than `t`. Every plane is treated alike; a subsampled
    /// chroma sample that holds or resets does so for the pixels it
    /// covers, which is the right granularity for the noise it carries.
    pub fn planes(cur: &mut [u8], prev: &[u8], t: u8) {
        debug_assert_eq!(cur.len(), prev.len());
        for (c, &p) in cur.iter_mut().zip(prev) {
            if c.abs_diff(p) < t {
                *c = p;
            }
        }
    }

    /// Per-sample threshold for planar input from the RGB L1 threshold:
    /// a luma step of one moves every RGB channel by about one, so an
    /// L1 budget of `t` over three channels is roughly `t / 3` per sample.
    pub fn plane_threshold(t: u32) -> u8 {
        t.div_ceil(3).min(255) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hold_rgba_keeps_small_moves_and_passes_large() {
        let prev = [100u8, 100, 100, 255, 10, 10, 10, 255, 50, 50, 50, 255];
        let mut cur = [103u8, 99, 101, 255, 40, 10, 10, 255, 52, 50, 50, 0];
        hold::rgba(&mut cur, &prev, 8);
        assert_eq!(&cur[0..4], &prev[0..4], "L1 5 < 8 holds");
        assert_eq!(&cur[4..8], &[40, 10, 10, 255], "L1 30 passes through");
        assert_eq!(&cur[8..12], &[52, 50, 50, 0], "alpha change never holds");
    }

    #[test]
    fn hold_planes_and_threshold() {
        let prev = [100u8, 100, 100];
        let mut cur = [102u8, 97, 104];
        hold::planes(&mut cur, &prev, 3);
        assert_eq!(cur, [100, 97, 104]);
        assert_eq!(hold::plane_threshold(8), 3);
        assert_eq!(hold::plane_threshold(12), 4);
        assert_eq!(hold::plane_threshold(0), 0);
    }

    #[test]
    fn header_basic() {
        let m = parse_y4m_header("YUV4MPEG2 W640 H360 F30000:1001 Ip A1:1 C420jpeg\n").unwrap();
        assert_eq!((m.width, m.height), (640, 360));
        assert_eq!((m.fps_num, m.fps_den), (30000, 1001));
        assert_eq!(m.chroma, Some(Chroma::C420));
    }

    #[test]
    fn header_rejects_high_depth() {
        assert!(parse_y4m_header("YUV4MPEG2 W64 H36 F25:1 C420p10\n").is_err());
    }

    #[test]
    fn frame_sizes() {
        assert_eq!(Chroma::C420.frame_bytes(640, 360), 640 * 360 * 3 / 2);
        assert_eq!(Chroma::C444.frame_bytes(64, 36), 64 * 36 * 3);
        assert_eq!(Chroma::C420.frame_bytes(3, 3), 9 + 2 * 4);
    }
}
