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
    /// per-frame palette indices as encoded, before compositing — the
    /// composited view above cannot show transparency, since it keeps
    /// whatever the canvas already held
    raw: Vec<RawFrame>,
    loop_ext: bool,
}

/// One image block exactly as it was encoded: its sub-rectangle, its
/// palette indices, and the transparent index in force for it.
struct RawFrame {
    rect: (usize, usize, usize, usize), // x0, y0, w, h
    px: Vec<u8>,
    transparent: Option<u8>,
}

impl RawFrame {
    /// Indices of a frame that covers the whole canvas (what every frame
    /// looks like once a clip carries any transparency).
    fn full(&self, width: usize, height: usize) -> &[u8] {
        assert_eq!(
            self.rect,
            (0, 0, width, height),
            "expected a full-canvas frame"
        );
        &self.px
    }
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
    let mut raw = Vec::new();
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
                raw.push(RawFrame {
                    rect: (x0, y0, sw, sh),
                    px: pixels.clone(),
                    transparent,
                });
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
        raw,
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

/// Transparency in *any* frame must survive, including when other frames in
/// the same clip are fully opaque — those are packed down to RGB in pass 1,
/// and the packing must neither lose a transparent pixel nor stop the clip
/// from switching to whole-frame, restore-to-background encoding.
#[test]
fn mixed_alpha_frames_keep_transparency() {
    let (w, h) = (16usize, 16usize);
    let a = [10u8, 200, 10, 255];
    let b = [200u8, 10, 10, 255];
    let hole = [0u8, 0, 0, 0];
    let mut raw = Vec::new();
    // frame 0: fully opaque (packs to RGB)
    raw.extend(std::iter::repeat_n(a, w * h).flatten());
    // frame 1: same, but with a transparent 4x4 hole (stays RGBA)
    for y in 0..h {
        for x in 0..w {
            let c = if (4..8).contains(&x) && (4..8).contains(&y) {
                hole
            } else {
                a
            };
            raw.extend_from_slice(&c);
        }
    }
    // frame 2: fully opaque, different color (packs to RGB)
    raw.extend(std::iter::repeat_n(b, w * h).flatten());

    let gif = run_weft(&["--size", "16x16", "--fps", "10"], &raw);
    let dec = decode_gif(&gif);
    assert_eq!(dec.raw.len(), 3, "three distinct frames");

    let trans = dec.raw[1].transparent.expect("transparent index declared");
    // frames from opaque sources carry no transparent pixel at all
    for f in [0usize, 2] {
        let px = dec.raw[f].full(w, h);
        assert!(
            px.iter().all(|&p| Some(p) != dec.raw[f].transparent),
            "frame {f} came from an opaque source, so nothing may be transparent"
        );
        assert!(
            px.iter().all(|&p| p == px[0]),
            "frame {f} is one flat color"
        );
    }
    // and the two opaque frames kept their distinct colors
    let opaque = dec.raw[0].full(w, h)[0];
    assert_ne!(opaque, dec.raw[2].full(w, h)[0]);
    // the alpha frame's hole is transparent, everything else matches frame 0
    for y in 0..h {
        for x in 0..w {
            let p = dec.raw[1].full(w, h)[y * w + x];
            if (4..8).contains(&x) && (4..8).contains(&y) {
                assert_eq!(p, trans, "hole pixel ({x},{y})");
            } else {
                assert_eq!(p, opaque, "opaque pixel ({x},{y})");
            }
        }
    }
}

/// Alpha is a binary test at 128, so any pixel at or above it is opaque and
/// the frame is packable — the stored alpha byte itself is never read.
#[test]
fn partial_alpha_at_or_above_threshold_is_opaque() {
    let (w, h) = (16usize, 16usize);
    let mut raw = Vec::new();
    for i in 0..w * h {
        // every alpha from 128 to 255: all of them count as opaque, so the
        // frame packs to RGB and the stored alpha byte is never read back
        raw.extend_from_slice(&[10, 200, 10, 128 + (i / 2) as u8]);
    }
    let gif = run_weft(&["--size", "16x16"], &raw);
    let dec = decode_gif(&gif);
    assert_eq!(dec.raw.len(), 1);
    let f = &dec.raw[0];
    assert!(
        f.full(w, h).iter().all(|&p| Some(p) != f.transparent),
        "no pixel may decode as transparent"
    );
    assert_eq!(dec.frames[0].0[..3], [10, 200, 10]);
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

/// A source whose colors all fit in the palette has no quantization error
/// to hide, so every dither mode must reproduce it exactly — including
/// `bayer`, whose threshold offset would otherwise push closely spaced
/// grays onto their neighbors (the shape of any soft gradient).
#[test]
fn bayer_is_lossless_on_exact_palette() {
    let (w, h, n) = (32usize, 32usize, 4usize);
    let fg = [200u8, 30, 30, 255];
    let mut raw = Vec::new();
    let mut expect: Vec<Vec<u8>> = Vec::new();
    for f in 0..n {
        let mut frame = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                // vertical gradient in steps of 4 — closer together than
                // the Bayer mask's +/-8 offset — plus a moving box
                let v = (y as u8) * 4;
                let inside = x >= f * 4 && x < f * 4 + 8 && (8..16).contains(&y);
                let c = if inside { fg } else { [v, v, v, 255] };
                frame.extend_from_slice(&c);
            }
        }
        expect.push(
            frame
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect(),
        );
        raw.extend_from_slice(&frame);
    }
    let gif = run_weft(
        &["--size", "32x32", "--fps", "10", "--dither", "bayer"],
        &raw,
    );
    let dec = decode_gif(&gif);
    assert_eq!(dec.frames.len(), n, "one visible frame per input frame");
    for (i, ((got, _), want)) in dec.frames.iter().zip(&expect).enumerate() {
        assert_eq!(got, want, "frame {i} was dithered despite an exact palette");
    }
}

#[test]
fn hold_folds_noisy_static_frames() {
    // A static solid frame with 0/1 per-channel noise re-rolled each
    // frame, then a genuinely different frame. Without --hold the noise
    // keeps every frame distinct; with it, the adaptive window (sized
    // from the noise itself) holds the noisy frames at the first frame's
    // values and they fold into one delay.
    let (w, h) = (32usize, 32usize);
    let mut raw = Vec::new();
    let mut seed = 0x9E37_79B9u32;
    for f in 0..4 {
        for _ in 0..w * h {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = |k: u32| ((seed >> k) % 2) as u8; // 0, 1
            raw.extend_from_slice(&[99 + n(3), 149 + n(11), 199 + n(19), 255]);
            let _ = f;
        }
    }
    raw.extend(std::iter::repeat_n([200u8, 10, 10, 255], w * h).flatten());
    let plain = decode_gif(&run_weft(&["--size", "32x32", "--fps", "10"], &raw));
    assert!(plain.frames.len() > 2, "noise should keep frames distinct");
    let held = decode_gif(&run_weft(
        &["--size", "32x32", "--fps", "10", "--hold", "8"],
        &raw,
    ));
    assert_eq!(held.frames.len(), 2, "held frames fold into one");
    assert_eq!(held.frames[0].1, 40, "4 x 10cs");
    assert_eq!(
        held.frames[1].0[..3],
        [200, 10, 10],
        "large change passes through"
    );
}

#[test]
fn smooth_flattens_grain_and_keeps_edges() {
    // Two halves of different colours with +-2 grain on every pixel and
    // a fresh grain roll per frame. --smooth alone makes each frame flat
    // (so the two frames become identical and fold); the edge between
    // the halves must stay exactly where it was.
    let (w, h) = (32usize, 16usize);
    let mut raw = Vec::new();
    let mut seed = 0x1234_5678u32;
    for _ in 0..2 {
        for y in 0..h {
            for x in 0..w {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let g = ((seed >> 24) % 5) as u8; // 0..4 -> -2..+2
                let base: [u8; 3] = if x < 16 {
                    [60, 120, 180]
                } else {
                    [200, 80, 40]
                };
                raw.extend_from_slice(&[base[0] + g - 2, base[1] + g - 2, base[2] + g - 2, 255]);
                let _ = y;
            }
        }
    }
    let plain = decode_gif(&run_weft(&["--size", "32x16", "--fps", "10"], &raw));
    assert_eq!(plain.frames.len(), 2, "grain keeps the frames distinct");
    // smoothing alone: every pixel within 1 of its fill colour (the 5x5
    // mean of +-2 grain), edge column exact
    let smooth = decode_gif(&run_weft(
        &["--size", "32x16", "--fps", "10", "--smooth", "24"],
        &raw,
    ));
    for (fi, (f, _)) in smooth.frames.iter().enumerate() {
        for y in 0..h {
            for x in 0..w {
                let px = &f[(y * w + x) * 3..(y * w + x) * 3 + 3];
                let want: [u8; 3] = if x < 16 {
                    [60, 120, 180]
                } else {
                    [200, 80, 40]
                };
                for c in 0..3 {
                    assert!(
                        px[c].abs_diff(want[c]) <= 1,
                        "frame {fi} ({x},{y}): {px:?} vs {want:?}"
                    );
                }
            }
        }
    }
    // with the hold on top, the residual +-1 is inside the window and the
    // two frames fold into one
    let both = decode_gif(&run_weft(
        &[
            "--size", "32x16", "--fps", "10", "--smooth", "24", "--hold", "8",
        ],
        &raw,
    ));
    assert_eq!(both.frames.len(), 1, "smoothed + held frames fold");
}
