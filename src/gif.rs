//! GIF stream assembly: inter-frame delta (transparency + bbox crop),
//! per-frame body encoding, and the sequential muxer.

use crate::lzw::LzwEncoder;

pub struct EncodedFrame {
    /// Delay in centiseconds (patched at mux time when duplicate frames merge).
    pub delay_cs: u32,
    /// Image descriptor + LZW data; empty when the frame is identical to the
    /// previous one (muxer folds its delay into the predecessor).
    pub body: Vec<u8>,
    pub disposal: u8,
}

pub const DISPOSAL_NONE: u8 = 1;
pub const DISPOSAL_BACKGROUND: u8 = 2;

/// Encode one indexed frame relative to the previous full indexed frame.
///
/// With disposal "none" and no source alpha, the canvas after frame i-1 is
/// exactly the indexed frame i-1, so the delta for frame i only needs frame
/// i-1 — every frame can encode in parallel. `prev == None` means keyframe
/// (first frame, or delta disabled because the source has alpha).
pub fn encode_frame(
    idx: &[u8],
    prev: Option<&[u8]>,
    w: usize,
    h: usize,
    trans_idx: u8,
    min_code_size: u8,
    delay_cs: u32,
    disposal: u8,
    enc: &mut LzwEncoder,
) -> EncodedFrame {
    let mut body = Vec::new();

    let (x0, y0, x1, y1, sub) = match prev {
        None => (0, 0, w - 1, h - 1, None),
        Some(prev) => {
            // Row-level diff first (fast memcmp path), then column bounds.
            let mut y0 = None;
            let mut y1 = 0usize;
            for y in 0..h {
                if idx[y * w..(y + 1) * w] != prev[y * w..(y + 1) * w] {
                    if y0.is_none() {
                        y0 = Some(y);
                    }
                    y1 = y;
                }
            }
            let Some(y0) = y0 else {
                // identical frame: fold into predecessor at mux time
                return EncodedFrame { delay_cs, body, disposal };
            };
            let mut x0 = w;
            let mut x1 = 0usize;
            for y in y0..=y1 {
                let a = &idx[y * w..(y + 1) * w];
                let b = &prev[y * w..(y + 1) * w];
                if let Some(first) = a.iter().zip(b).position(|(p, q)| p != q) {
                    x0 = x0.min(first);
                    let last = w - 1 - a.iter().zip(b.iter()).rev().position(|(p, q)| p != q).unwrap();
                    x1 = x1.max(last);
                }
            }
            // Build sub-image with unchanged pixels replaced by transparency.
            let sw = x1 - x0 + 1;
            let sh = y1 - y0 + 1;
            let mut sub = Vec::with_capacity(sw * sh);
            for y in y0..=y1 {
                let a = &idx[y * w + x0..y * w + x1 + 1];
                let b = &prev[y * w + x0..y * w + x1 + 1];
                sub.extend(a.iter().zip(b).map(|(&p, &q)| if p == q { trans_idx } else { p }));
            }
            (x0, y0, x1, y1, Some(sub))
        }
    };

    let sw = x1 - x0 + 1;
    let sh = y1 - y0 + 1;
    // image descriptor
    body.push(0x2C);
    body.extend_from_slice(&(x0 as u16).to_le_bytes());
    body.extend_from_slice(&(y0 as u16).to_le_bytes());
    body.extend_from_slice(&(sw as u16).to_le_bytes());
    body.extend_from_slice(&(sh as u16).to_le_bytes());
    body.push(0); // no local color table, not interlaced

    match &sub {
        Some(s) => enc.encode(min_code_size, s, &mut body),
        None => enc.encode(min_code_size, idx, &mut body),
    }

    EncodedFrame { delay_cs, body, disposal }
}

pub struct MuxParams<'a> {
    pub width: usize,
    pub height: usize,
    pub colors: &'a [[u8; 3]],
    pub trans_idx: u8,
    /// GCT size as a power of two >= colors + 1 (transparent slot).
    pub gct_bits: u8,
    /// None = no NETSCAPE loop extension; Some(0) = loop forever.
    pub loop_count: Option<u16>,
}

/// Write the full GIF stream. Consecutive empty-body frames fold their
/// delays into the previous visible frame.
pub fn mux(params: &MuxParams, frames: &[EncodedFrame], out: &mut Vec<u8>) {
    out.extend_from_slice(b"GIF89a");
    out.extend_from_slice(&(params.width as u16).to_le_bytes());
    out.extend_from_slice(&(params.height as u16).to_le_bytes());
    out.push(0x80 | 0x70 | (params.gct_bits - 1)); // GCT flag, 8-bit color res, size
    out.push(0); // background color index
    out.push(0); // aspect
    let gct_len = 1usize << params.gct_bits;
    for i in 0..gct_len {
        let c = params.colors.get(i).copied().unwrap_or([0, 0, 0]);
        out.extend_from_slice(&c);
    }
    if let Some(loops) = params.loop_count {
        out.extend_from_slice(&[0x21, 0xFF, 0x0B]);
        out.extend_from_slice(b"NETSCAPE2.0");
        out.extend_from_slice(&[0x03, 0x01]);
        out.extend_from_slice(&loops.to_le_bytes());
        out.push(0);
    }

    // Fold empty frames' delays forward into the previous visible frame.
    let mut delays: Vec<u32> = Vec::with_capacity(frames.len());
    let mut last_visible: Option<usize> = None;
    for f in frames {
        if f.body.is_empty() {
            if let Some(i) = last_visible {
                delays[i] = (delays[i] + f.delay_cs).min(u16::MAX as u32);
                delays.push(0);
                continue;
            }
        }
        last_visible = Some(delays.len());
        delays.push(f.delay_cs.min(u16::MAX as u32));
    }

    for (f, &delay) in frames.iter().zip(&delays) {
        if f.body.is_empty() {
            continue;
        }
        // graphic control extension
        out.extend_from_slice(&[0x21, 0xF9, 0x04, (f.disposal << 2) | 0x01]);
        out.extend_from_slice(&(delay as u16).to_le_bytes());
        out.push(params.trans_idx);
        out.push(0);
        out.extend_from_slice(&f.body);
    }
    out.push(0x3B);
}

/// Per-frame delay in centiseconds via error-free accumulation:
/// delay_i = round(100*(i+1)*den/num) - round(100*i*den/num).
pub fn frame_delays(n: usize, fps_num: u32, fps_den: u32) -> Vec<u32> {
    let num = fps_num as u64;
    let den = fps_den as u64;
    let at = |i: u64| (100 * i * den + num / 2) / num;
    (0..n as u64).map(|i| (at(i + 1) - at(i)) as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delays_accumulate_exactly() {
        let d = frame_delays(30, 30, 1);
        assert_eq!(d.iter().sum::<u32>(), 100);
        let d = frame_delays(30000, 30000, 1001);
        assert_eq!(d.iter().sum::<u32>(), 100_100);
        for &x in &frame_delays(100, 50, 1) {
            assert_eq!(x, 2);
        }
    }

    #[test]
    fn identical_frame_returns_empty_body() {
        let a = vec![5u8; 16];
        let mut enc = LzwEncoder::default();
        let f = encode_frame(&a, Some(&a), 4, 4, 255, 8, 3, DISPOSAL_NONE, &mut enc);
        assert!(f.body.is_empty());
        assert_eq!(f.delay_cs, 3);
    }

    #[test]
    fn bbox_is_tight() {
        let w = 8;
        let h = 8;
        let prev = vec![0u8; w * h];
        let mut cur = prev.clone();
        cur[3 * w + 2] = 9; // single changed pixel at (2,3)
        let mut enc = LzwEncoder::default();
        let f = encode_frame(&cur, Some(&prev), w, h, 255, 8, 3, DISPOSAL_NONE, &mut enc);
        // descriptor: 2C x0 y0 w h
        assert_eq!(f.body[0], 0x2C);
        assert_eq!(u16::from_le_bytes([f.body[1], f.body[2]]), 2);
        assert_eq!(u16::from_le_bytes([f.body[3], f.body[4]]), 3);
        assert_eq!(u16::from_le_bytes([f.body[5], f.body[6]]), 1);
        assert_eq!(u16::from_le_bytes([f.body[7], f.body[8]]), 1);
    }
}
