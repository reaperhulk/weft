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
/// the deviation bound. These are the scalar reference kernels; the
/// reader uses the SIMD versions in `simdops` (which also mirror the
/// result into the reference buffer) and falls back to these for tails.
pub mod hold {
    /// Per-sample threshold for planar input from the RGB L1 threshold:
    /// a luma step of one moves every RGB channel by about one, so an
    /// L1 budget of `t` over three channels is roughly `t / 3` per sample.
    pub fn plane_threshold(t: u32) -> u8 {
        t.div_ceil(3).min(255) as u8
    }

    /// Adaptive window from a histogram of per-pixel L1 change between
    /// consecutive raw frames (bins 0..=255, saturating): the larger of
    /// 2.5x the median change and the 75th-percentile change, clamped to
    /// [4, cap] (4 measured clean on every clip, including a slow zoom). On grainy content the upper quartile is the grain level
    /// (a scanned cartoon lands near 8, where the fixed window was
    /// tuned); on a clean source both statistics are ~0 and the window
    /// closes to the floor — a fixed window sized for grain would only
    /// lag real slow motion there and smear edges (measured: a slow zoom
    /// at a fixed 12 lost 3 dB and doubled every edge). Widespread motion
    /// raises both statistics, and then `cap` — the user's `--hold N` —
    /// bounds the damage; the 90th percentile was tried and opens the
    /// window on motion too readily.
    pub fn adaptive_threshold(hist: &[u32; 256], cap: u32) -> u32 {
        let total: u64 = hist.iter().map(|&c| c as u64).sum();
        let quantile = |num: u64, den: u64| -> u32 {
            let mut acc = 0u64;
            for (i, &c) in hist.iter().enumerate() {
                acc += c as u64;
                if acc * den >= total * num {
                    return i as u32;
                }
            }
            255
        };
        let median = quantile(1, 2);
        let q75 = quantile(3, 4);
        ((median * 5).div_ceil(2)).max(q75).clamp(4, cap.max(4))
    }

    /// The bound on how far a held pixel may sit from its reference:
    /// the mean-centred test alone would let the output lag a slow drift
    /// without limit (measured: up to 45/765 on a lighting change).
    pub fn max_deviation(t: u32) -> u32 {
        t + t / 2
    }

    /// Fixed-point format of the running mean: 8.7 in i16 (255 << 7 =
    /// 32640 fits, and so does any difference of two such values), so
    /// the update `m += (cur - m) >> 3` is exact 16-bit arithmetic.
    pub const MEAN_SHIFT: u32 = 7;
    /// Update rate of the running mean: 1/8 per frame.
    pub const MEAN_RATE: u32 = 3;

    #[inline(always)]
    pub fn mean_round(m: i16) -> u8 {
        ((m as i32 + (1 << (MEAN_SHIFT - 1))) >> MEAN_SHIFT) as u8
    }

    /// Mean-centred hold, RGBA. `mean` holds one 8.7 fixed-point value per
    /// byte of `cur` (the alpha lane is carried but unused). A pixel is
    /// held when its L1 distance (over all four bytes: an alpha change
    /// counts like any other) to the rounded running mean is below `t`
    /// and to the reference below `tmax`;
    /// then the output is the reference and the mean tracks the input at
    /// 1/8 per frame. Otherwise the pixel passes through and becomes the
    /// new reference and mean. Centring the window on the mean of the
    /// noise instead of on whatever sample happened to reset last cuts
    /// needless resets (measured 24% -> 22% of pixels per frame on a
    /// grainy clip, -7% file size at equal PSNR).
    pub fn rgba_mean(cur: &mut [u8], prev: &[u8], mean: &mut [i16], t: u32, tmax: u32) {
        debug_assert_eq!(cur.len(), prev.len());
        debug_assert_eq!(cur.len(), mean.len());
        for ((c, p), m) in cur
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(prev.as_chunks::<4>().0)
            .zip(mean.as_chunks_mut::<4>().0)
        {
            let dm: u32 = (0..4).map(|k| c[k].abs_diff(mean_round(m[k])) as u32).sum();
            let dr: u32 = (0..4).map(|k| c[k].abs_diff(p[k]) as u32).sum();
            if dm < t && dr < tmax {
                for k in 0..4 {
                    let d = ((c[k] as i16) << MEAN_SHIFT) - m[k];
                    m[k] += d >> MEAN_RATE;
                }
                *c = *p;
            } else {
                for k in 0..4 {
                    m[k] = (c[k] as i16) << MEAN_SHIFT;
                }
            }
        }
    }

    /// Mean-centred hold for planar samples (see `rgba_mean`).
    pub fn planes_mean(cur: &mut [u8], prev: &[u8], mean: &mut [i16], t: u8, tmax: u8) {
        for ((c, &p), m) in cur.iter_mut().zip(prev).zip(mean.iter_mut()) {
            if c.abs_diff(mean_round(*m)) < t && c.abs_diff(p) < tmax {
                let d = ((*c as i16) << MEAN_SHIFT) - *m;
                *m += d >> MEAN_RATE;
                *c = p;
            } else {
                *m = (*c as i16) << MEAN_SHIFT;
            }
        }
    }
}

/// Plane geometry of a stored Y4M frame: (byte offset, width, height)
/// per plane, in buffer order.
pub fn planes(chroma: Chroma, w: usize, h: usize) -> Vec<(usize, usize, usize)> {
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    match chroma {
        Chroma::Mono => vec![(0, w, h)],
        Chroma::C420 => vec![(0, w, h), (w * h, cw, ch), (w * h + cw * ch, cw, ch)],
        Chroma::C422 => vec![(0, w, h), (w * h, cw, h), (w * h + cw * h, cw, h)],
        Chroma::C444 => vec![(0, w, h), (w * h, w, h), (2 * w * h, w, h)],
    }
}

/// Spatial grain filter for `--smooth`: a range-limited 5x5 box filter.
/// Each pixel becomes the mean of the window neighbours whose distance to
/// it is below the threshold, so film grain and codec noise average out
/// inside a fill while anything across an edge is excluded and outlines
/// stay crisp. Runs before the hold: with the grain gone, far fewer
/// pixels exceed the hold window, and the palette sees clean fills.
/// (Measured with hold: -6..-11% file size with PSNR up 0.1-0.2 dB on a
/// cartoon corpus; grain-free content is untouched.)
///
/// The frame is padded by `PAD` replicated pixels on each side into a
/// scratch buffer, so every window load is in bounds and the source can
/// be overwritten in place. Frames are independent, so the pipeline runs
/// whole frames on a small pool (see main.rs); within a frame the rows
/// run serially.
pub mod smooth {

    pub const RADIUS: usize = 2;
    pub const PAD: usize = RADIUS;
    pub const WIN: usize = 2 * RADIUS + 1;

    /// Replicate-pad a plane of `bpp`-byte samples into `out`
    /// ((w + 2*PAD) x (h + 2*PAD)).
    pub fn pad(src: &[u8], w: usize, h: usize, bpp: usize, out: &mut Vec<u8>) {
        let pw = w + 2 * PAD;
        let ph = h + 2 * PAD;
        out.resize(pw * ph * bpp, 0);
        let pad_row = |(py, row): (usize, &mut [u8])| {
            let sy = py.saturating_sub(PAD).min(h - 1);
            let srow = &src[sy * w * bpp..(sy + 1) * w * bpp];
            row[PAD * bpp..(PAD + w) * bpp].copy_from_slice(srow);
            for i in 0..PAD {
                row[i * bpp..(i + 1) * bpp].copy_from_slice(&srow[..bpp]);
                row[(PAD + w + i) * bpp..(PAD + w + i + 1) * bpp]
                    .copy_from_slice(&srow[(w - 1) * bpp..]);
            }
        };
        out.chunks_mut(pw * bpp).enumerate().for_each(pad_row);
    }

    /// Scalar reference for one output row of RGBA: `padded` is the
    /// padded frame, `y` the output row, `out` its w pixels.
    #[cfg(test)]
    pub fn rgba_row(padded: &[u8], w: usize, y: usize, s: u32, out: &mut [u8]) {
        let pw = w + 2 * PAD;
        for (x, o) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let ci = ((y + PAD) * pw + x + PAD) * 4;
            let c = &padded[ci..ci + 4];
            let mut acc = [0u32; 3];
            let mut cnt = 0u32;
            for dy in 0..WIN {
                for dx in 0..WIN {
                    let ni = ((y + dy) * pw + x + dx) * 4;
                    let n = &padded[ni..ni + 4];
                    let d: u32 = (0..4).map(|k| n[k].abs_diff(c[k]) as u32).sum();
                    if d < s {
                        acc[0] += n[0] as u32;
                        acc[1] += n[1] as u32;
                        acc[2] += n[2] as u32;
                        cnt += 1;
                    }
                }
            }
            for k in 0..3 {
                o[k] = ((acc[k] + cnt / 2) / cnt) as u8;
            }
            o[3] = c[3];
        }
    }

    /// Scalar reference for one output row of a plane.
    #[cfg(test)]
    pub fn plane_row(padded: &[u8], w: usize, y: usize, t: u8, out: &mut [u8]) {
        let pw = w + 2 * PAD;
        for (x, o) in out.iter_mut().enumerate() {
            let c = padded[(y + PAD) * pw + x + PAD];
            let mut acc = 0u32;
            let mut cnt = 0u32;
            for dy in 0..WIN {
                for dx in 0..WIN {
                    let n = padded[(y + dy) * pw + x + dx];
                    if n.abs_diff(c) < t {
                        acc += n as u32;
                        cnt += 1;
                    }
                }
            }
            *o = ((acc + cnt / 2) / cnt) as u8;
        }
    }

    /// Smooth an RGBA frame in place.
    pub fn rgba(
        level: fearless_simd::Level,
        frame: &mut [u8],
        w: usize,
        h: usize,
        s: u32,
        scratch: &mut Vec<u8>,
    ) {
        pad(frame, w, h, 4, scratch);
        let padded: &[u8] = scratch;
        for (y, row) in frame.chunks_mut(w * 4).enumerate() {
            crate::simdops::smooth_rgba_row(level, padded, w, y, s, row);
        }
    }

    /// Smooth every plane of a Y4M frame in place.
    pub fn planes(
        level: fearless_simd::Level,
        frame: &mut [u8],
        geom: &[(usize, usize, usize)],
        t: u8,
        scratch: &mut Vec<u8>,
    ) {
        for &(off, pw, ph) in geom {
            let plane = &mut frame[off..off + pw * ph];
            pad(plane, pw, ph, 1, scratch);
            let padded: &[u8] = scratch;
            for (y, row) in plane.chunks_mut(pw).enumerate() {
                crate::simdops::smooth_plane_row(level, padded, pw, y, t, row);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hold_mean_centres_window_and_bounds_lag() {
        // reference at 100, mean drifts toward 106 while held; a pixel at
        // 107 is within t=8 of the mean but only held while within tmax of
        // the reference
        let prev = [100u8, 100, 100, 255];
        let mut mean = [100i16 << hold::MEAN_SHIFT; 4];
        mean[3] = 255 << hold::MEAN_SHIFT;
        let mut cur = [103u8, 102, 102, 255];
        hold::rgba_mean(&mut cur, &prev, &mut mean, 8, 12);
        assert_eq!(cur, prev, "L1 7 < 8 holds");
        assert!(
            hold::mean_round(mean[0]) == 100,
            "mean moves 1/8 of 3: rounds to 100"
        );
        let mut m = mean;
        for _ in 0..40 {
            let mut c = [106u8, 106, 106, 255];
            hold::rgba_mean(&mut c, &prev, &mut m, 20, 40);
            assert_eq!(c, prev);
        }
        assert_eq!(hold::mean_round(m[0]), 106, "mean converges to the input");
        // now within t of the mean but beyond tmax of the reference: reset
        let mut c = [109u8, 109, 109, 255];
        hold::rgba_mean(&mut c, &prev, &mut m, 20, 24);
        assert_eq!(c, [109, 109, 109, 255]);
        assert_eq!(hold::mean_round(m[0]), 109, "reset re-seeds the mean");
        assert_eq!(hold::max_deviation(8), 12);
    }

    #[test]
    fn smooth_removes_grain_keeps_edges() {
        // 12x8 frame: left half 100, right half 160, one +3 speck at (3,5)
        let (w, h) = (12usize, 8usize);
        let mut f = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let v = if x < 6 { 100 } else { 160 };
                let v = if (x, y) == (3, 5) { v + 3 } else { v };
                f[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        let mut pad = Vec::new();
        smooth::pad(&f, w, h, 4, &mut pad);
        let mut out = vec![0u8; w * 4];
        for y in 0..h {
            smooth::rgba_row(&pad, w, y, 24, &mut out);
            for x in 0..w {
                let want = if x < 6 { 100 } else { 160 };
                assert_eq!(out[x * 4], want, "({x},{y})");
            }
        }
    }

    #[test]
    fn plane_geometry() {
        assert_eq!(
            planes(Chroma::C420, 5, 3),
            vec![(0, 5, 3), (15, 3, 2), (21, 3, 2)]
        );
        assert_eq!(
            planes(Chroma::C444, 4, 2),
            vec![(0, 4, 2), (8, 4, 2), (16, 4, 2)]
        );
        assert_eq!(planes(Chroma::Mono, 4, 2), vec![(0, 4, 2)]);
    }

    #[test]
    fn adaptive_threshold_tracks_noise() {
        let mut h = [0u32; 256];
        h[1] = 60;
        h[2] = 30;
        h[4] = 10; // upper quartile at 2: max(2.5*1 -> 3, 2) -> floor 4
        assert_eq!(hold::adaptive_threshold(&h, 12), 4);
        let mut h = [0u32; 256];
        h[1] = 60;
        h[6] = 40; // grain: upper quartile at 6
        assert_eq!(hold::adaptive_threshold(&h, 12), 6);
        let mut h = [0u32; 256];
        h[4] = 100;
        assert_eq!(hold::adaptive_threshold(&h, 12), 10);
        assert_eq!(hold::adaptive_threshold(&h, 8), 8, "capped by --hold");
        let mut h = [0u32; 256];
        h[0] = 95;
        h[40] = 5; // a little motion does not open the window
        assert_eq!(hold::adaptive_threshold(&h, 12), 4);
        let h = [0u32; 256];
        assert_eq!(hold::adaptive_threshold(&h, 12), 4, "empty: floor");
    }

    #[test]
    fn hold_planes_and_threshold() {
        let prev = [100u8, 100, 100];
        let mut cur = [102u8, 97, 104];
        let mut mean = [100i16 << hold::MEAN_SHIFT; 3];
        hold::planes_mean(&mut cur, &prev, &mut mean, 3, 5);
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
