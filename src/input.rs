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
        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        match self {
            Chroma::C420 => w * h + 2 * cw * ch,
            Chroma::C422 => w * h + 2 * cw * h,
            Chroma::C444 => 3 * w * h,
            Chroma::Mono => w * h,
        }
    }
}

/// A stored input frame: either raw RGBA bytes or raw Y4M planes.
/// Y4M frames are kept in their native (smaller) form and converted to RGBA
/// on the fly in each parallel pass, trading a cheap reconversion for a
/// much smaller resident set.
pub enum Frame {
    Rgba(Vec<u8>),
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

/// Read all Y4M frames after the stream header.
pub fn read_y4m_frames(
    r: &mut impl BufRead,
    meta: &VideoIn,
) -> io::Result<Vec<Frame>> {
    let chroma = meta.chroma.unwrap();
    let fsize = chroma.frame_bytes(meta.width, meta.height);
    let mut frames = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        let n = r.read_line(&mut line)?;
        if n == 0 {
            break; // clean EOF
        }
        if !line.starts_with("FRAME") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "y4m: expected FRAME marker",
            ));
        }
        let mut buf = vec![0u8; fsize];
        r.read_exact(&mut buf)?;
        frames.push(Frame::Yuv(buf));
    }
    Ok(frames)
}

/// Read raw RGBA frames until EOF. A trailing partial frame is an error.
pub fn read_rgba_frames(r: &mut impl Read, w: usize, h: usize) -> io::Result<Vec<Frame>> {
    let fsize = w * h * 4;
    let mut frames = Vec::new();
    loop {
        let mut buf = vec![0u8; fsize];
        let mut got = 0usize;
        while got < fsize {
            let n = r.read(&mut buf[got..])?;
            if n == 0 {
                break;
            }
            got += n;
        }
        if got == 0 {
            break;
        }
        if got < fsize {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("raw rgba: truncated frame ({got} of {fsize} bytes)"),
            ));
        }
        frames.push(Frame::Rgba(buf));
    }
    Ok(frames)
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
