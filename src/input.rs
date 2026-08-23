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

/// A decoded input frame: either raw RGBA bytes or raw Y4M planes.
/// Y4M frames are kept in their native (smaller) form and converted to RGBA
/// on the fly in each parallel pass, trading a cheap reconversion for a
/// much smaller resident set.
pub enum Frame {
    Rgba(Vec<u8>),
    Yuv(Vec<u8>),
}

/// A frame as buffered between the histogram and quantize passes,
/// LZ4-compressed whenever that wins. Frames are touched exactly twice —
/// streamed through the histogram, then through quantization — and sit idle
/// for the entire palette build in between, so the resident set between the
/// passes shrinks to roughly the LZ4 size of the input while every consumer
/// still sees plain raw bytes (the interpretation — RGBA or Y4M planes — is
/// the stream's `chroma`, uniform across frames).
pub struct StoredFrame {
    data: Vec<u8>,
    /// Uncompressed length; `data` is an LZ4 block iff `data.len() < raw_len`
    /// (raw is kept whenever compression doesn't strictly shrink, so the
    /// comparison is unambiguous).
    raw_len: usize,
}

impl StoredFrame {
    pub fn pack(frame: Frame) -> StoredFrame {
        let (Frame::Rgba(raw) | Frame::Yuv(raw)) = frame;
        let data = lz4_flex::block::compress(&raw);
        if data.len() < raw.len() {
            StoredFrame {
                data,
                raw_len: raw.len(),
            }
        } else {
            StoredFrame {
                raw_len: raw.len(),
                data: raw,
            }
        }
    }

    /// The raw frame bytes, decompressing into `scratch` when needed.
    /// `scratch` is caller-owned so parallel consumers can reuse one
    /// allocation per worker.
    pub fn unpack<'a>(&'a self, scratch: &'a mut Vec<u8>) -> &'a [u8] {
        if self.data.len() == self.raw_len {
            return &self.data;
        }
        scratch.resize(self.raw_len, 0);
        let n = lz4_flex::block::decompress_into(&self.data, scratch)
            .expect("in-memory LZ4 frame corrupt");
        debug_assert_eq!(n, self.raw_len);
        &scratch[..n]
    }
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

/// Read exactly `n` bytes into a fresh buffer without pre-zeroing it
/// (via `take` + `read_to_end`, which fills spare capacity directly).
/// Returns None on immediate clean EOF; errors on a short read.
fn read_frame_buf(r: &mut impl Read, n: usize, what: &str) -> io::Result<Option<Vec<u8>>> {
    let mut buf = Vec::with_capacity(n);
    let got = r.take(n as u64).read_to_end(&mut buf)?;
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

/// Read the next Y4M frame (FRAME marker + planes), or None at EOF.
pub fn read_y4m_frame(r: &mut impl BufRead, meta: &VideoIn) -> io::Result<Option<Frame>> {
    let chroma = meta.chroma.unwrap();
    let fsize = chroma.frame_bytes(meta.width, meta.height);
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
    match read_frame_buf(r, fsize, "y4m")? {
        Some(buf) => Ok(Some(Frame::Yuv(buf))),
        None => Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "y4m: FRAME marker with no payload",
        )),
    }
}

/// Read the next raw RGBA frame, or None at EOF.
pub fn read_rgba_frame(r: &mut impl Read, w: usize, h: usize) -> io::Result<Option<Frame>> {
    Ok(read_frame_buf(r, w * h * 4, "raw rgba")?.map(Frame::Rgba))
}

#[cfg(test)]
mod tests {
    use super::*;

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
