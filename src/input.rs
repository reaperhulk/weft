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
