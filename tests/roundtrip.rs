//! End-to-end test: feed the weft binary raw RGBA frames, decode the GIF it
//! emits with an independent minimal decoder, and check the composited
//! canvas reproduces the input exactly (inputs use few distinct colors, so
//! the palette is lossless and sierra2 dithering is a no-op).

use std::io::Write;
use std::process::{Command, Stdio};

struct DecodedGif {
    width: usize,
    height: usize,
    /// composited full frames (palette-index canvases rendered to RGB)
    frames: Vec<(Vec<u8>, u32)>, // (rgb canvas, delay in cs)
    loop_ext: bool,
}

fn decode_gif(data: &[u8]) -> DecodedGif {
    assert_eq!(&data[..6], b"GIF89a", "header");
    let width = u16::from_le_bytes([data[6], data[7]]) as usize;
    let height = u16::from_le_bytes([data[8], data[9]]) as usize;
    let flags = data[10];
    assert!(flags & 0x80 != 0, "expected global color table");
    let gct_len = 2usize << (flags & 7);
    let mut pos = 13;
    let gct = &data[pos..pos + gct_len * 3];
    pos += gct_len * 3;

    let mut canvas: Vec<u8> = vec![0; width * height]; // palette indices
    let mut canvas_valid = false;
    let mut frames = Vec::new();
    let mut loop_ext = false;
    let mut delay = 0u32;
    let mut transparent: Option<u8> = None;

    loop {
        match data[pos] {
            0x3B => break, // trailer
            0x21 => {
                let label = data[pos + 1];
                pos += 2;
                if label == 0xF9 {
                    let bs = data[pos] as usize;
                    let block = &data[pos + 1..pos + 1 + bs];
                    delay = u16::from_le_bytes([block[1], block[2]]) as u32;
                    transparent = (block[0] & 1 != 0).then_some(block[3]);
                    pos += 1 + bs;
                    assert_eq!(data[pos], 0, "GCE terminator");
                    pos += 1;
                } else {
                    if label == 0xFF {
                        let bs = data[pos] as usize;
                        if &data[pos + 1..pos + 1 + bs] == b"NETSCAPE2.0" {
                            loop_ext = true;
                        }
                    }
                    // skip sub-blocks
                    loop {
                        let bs = data[pos] as usize;
                        pos += 1;
                        if bs == 0 {
                            break;
                        }
                        pos += bs;
                    }
                }
            }
            0x2C => {
                let x0 = u16::from_le_bytes([data[pos + 1], data[pos + 2]]) as usize;
                let y0 = u16::from_le_bytes([data[pos + 3], data[pos + 4]]) as usize;
                let sw = u16::from_le_bytes([data[pos + 5], data[pos + 6]]) as usize;
                let sh = u16::from_le_bytes([data[pos + 7], data[pos + 8]]) as usize;
                let lflags = data[pos + 9];
                assert_eq!(lflags & 0x80, 0, "no local color table expected");
                pos += 10;
                let mcs = data[pos];
                pos += 1;
                let mut payload = Vec::new();
                loop {
                    let bs = data[pos] as usize;
                    pos += 1;
                    if bs == 0 {
                        break;
                    }
                    payload.extend_from_slice(&data[pos..pos + bs]);
                    pos += bs;
                }
                let pixels = lzw_decode(mcs, &payload);
                assert_eq!(pixels.len(), sw * sh, "decoded pixel count");
                if !canvas_valid {
                    // first frame must cover everything opaque; treat
                    // uncovered as index 0
                    canvas_valid = true;
                }
                for yy in 0..sh {
                    for xx in 0..sw {
                        let p = pixels[yy * sw + xx];
                        if Some(p) != transparent {
                            canvas[(y0 + yy) * width + x0 + xx] = p;
                        }
                    }
                }
                let rgb: Vec<u8> = canvas
                    .iter()
                    .flat_map(|&i| {
                        let o = i as usize * 3;
                        [gct[o], gct[o + 1], gct[o + 2]]
                    })
                    .collect();
                frames.push((rgb, delay));
            }
            b => panic!("unexpected block 0x{b:02X} at {pos}"),
        }
    }
    DecodedGif {
        width,
        height,
        frames,
        loop_ext,
    }
}

fn lzw_decode(min_code_size: u8, mut bytes: &[u8]) -> Vec<u8> {
    let clear = 1usize << min_code_size;
    let eoi = clear + 1;
    let mut dict: Vec<Vec<u8>> = Vec::new();
    let reset = |dict: &mut Vec<Vec<u8>>| {
        dict.clear();
        for i in 0..clear {
            dict.push(vec![i as u8]);
        }
        dict.push(vec![]);
        dict.push(vec![]);
    };
    reset(&mut dict);
    let mut width = min_code_size as u32 + 1;
    let mut acc = 0u64;
    let mut nbits = 0u32;
    let mut out = Vec::new();
    let mut prev: Option<usize> = None;
    loop {
        while nbits < width {
            let (&b, rest) = bytes.split_first().expect("lzw data underrun");
            bytes = rest;
            acc |= (b as u64) << nbits;
            nbits += 8;
        }
        let code = (acc & ((1 << width) - 1)) as usize;
        acc >>= width;
        nbits -= width;
        if code == clear {
            reset(&mut dict);
            width = min_code_size as u32 + 1;
            prev = None;
            continue;
        }
        if code == eoi {
            break;
        }
        let entry = if code < dict.len() {
            dict[code].clone()
        } else {
            let p = &dict[prev.expect("KwKwK without prev")];
            let mut e = p.clone();
            e.push(p[0]);
            e
        };
        if let Some(p) = prev {
            let mut ne = dict[p].clone();
            ne.push(entry[0]);
            dict.push(ne);
            if dict.len() == (1 << width) && width < 12 {
                width += 1;
            }
        }
        prev = Some(code);
        out.extend_from_slice(&entry);
    }
    out
}

fn run_weft(args: &[&str], input: &[u8]) -> Vec<u8> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_weft"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn weft");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input)
        .expect("write stdin");
    let out = child.wait_with_output().expect("wait");
    assert!(out.status.success(), "weft exited with {:?}", out.status);
    out.stdout
}

/// Build simple RGBA frames: solid background with a moving opaque square.
fn synth_frames(w: usize, h: usize, n: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
    let bg = [40u8, 80, 120, 255];
    let fg = [200u8, 30, 30, 255];
    let mut raw = Vec::new();
    let mut rgbs = Vec::new();
    for f in 0..n {
        let mut frame = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let inside = x >= f && x < f + 8 && y >= f && y < f + 8;
                let c = if inside { fg } else { bg };
                frame.extend_from_slice(&c);
            }
        }
        rgbs.push(
            frame
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect(),
        );
        raw.extend_from_slice(&frame);
        rgbs.last().unwrap();
        let _ = f;
    }
    (raw, rgbs)
}

#[test]
fn roundtrip_exact_with_delta_frames() {
    let (w, h, n) = (64usize, 48usize, 12usize);
    let (raw, expect) = synth_frames(w, h, n);
    let gif = run_weft(&["--size", "64x48", "--fps", "20"], &raw);
    let dec = decode_gif(&gif);
    assert_eq!((dec.width, dec.height), (w, h));
    assert!(dec.loop_ext, "NETSCAPE loop extension expected by default");
    assert_eq!(dec.frames.len(), n, "one visible frame per input frame");
    for (i, ((got, delay), want)) in dec.frames.iter().zip(&expect).enumerate() {
        assert_eq!(*delay, 5, "20fps -> 5cs (frame {i})");
        assert_eq!(got, want, "frame {i} mismatch");
    }
}

#[test]
fn duplicate_frames_fold_delays() {
    let (w, h) = (32usize, 32usize);
    // 3 identical frames then a different one, then identical again
    let solid = |c: [u8; 4]| -> Vec<u8> { std::iter::repeat_n(c, w * h).flatten().collect() };
    let a = solid([10, 200, 10, 255]);
    let b = solid([200, 10, 10, 255]);
    let mut raw = Vec::new();
    for f in [&a, &a, &a, &b, &b] {
        raw.extend_from_slice(f);
    }
    let gif = run_weft(&["--size", "32x32", "--fps", "10"], &raw);
    let dec = decode_gif(&gif);
    assert_eq!(dec.frames.len(), 2, "dups must fold into predecessors");
    assert_eq!(dec.frames[0].1, 30, "3 frames x 10cs folded");
    assert_eq!(dec.frames[1].1, 20, "2 frames x 10cs folded");
    assert_eq!(dec.frames[0].0[..3], [10, 200, 10]);
    assert_eq!(dec.frames[1].0[..3], [200, 10, 10]);
}

#[test]
fn no_loop_flag_drops_extension() {
    let (w, h) = (16usize, 16usize);
    let raw: Vec<u8> = std::iter::repeat_n([1u8, 2, 3, 255], w * h)
        .flatten()
        .collect();
    let gif = run_weft(&["--size", "16x16", "--no-loop"], &raw);
    let dec = decode_gif(&gif);
    assert!(!dec.loop_ext);
    assert_eq!(dec.frames.len(), 1);
}

#[test]
fn y4m_smoke() {
    // tiny 4x4 y4m, 2 frames, C444 for exact color control
    let mut input = Vec::new();
    input.extend_from_slice(b"YUV4MPEG2 W4 H4 F25:1 Ip A1:1 C444\n");
    for _ in 0..2 {
        input.extend_from_slice(b"FRAME\n");
        input.extend_from_slice(&[128u8; 16]); // Y
        input.extend_from_slice(&[128u8; 16]); // U
        input.extend_from_slice(&[128u8; 16]); // V
    }
    let gif = run_weft(&[], &input);
    let dec = decode_gif(&gif);
    assert_eq!((dec.width, dec.height), (4, 4));
    assert_eq!(dec.frames.len(), 1, "identical frame folds");
    assert_eq!(dec.frames[0].1, 8, "2 frames at 25fps = 8cs");
    // Y=128,U=V=128 -> gray 130 (1.164*(128-16) = 130.4)
    let px = &dec.frames[0].0[..3];
    assert!(px.iter().all(|&c| (129..=131).contains(&c)), "{px:?}");
}
