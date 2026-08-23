//! weft: fast parallel GIF encoder.
//! Reads raw RGBA or yuv4mpegpipe from stdin, writes GIF to stdout.

#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod color;
mod dither;
mod gif;
mod input;
mod lzw;
mod oklab;
mod palette;

use dither::{Dither, Quantizer};
use input::{Frame, VideoIn};
use lzw::LzwEncoder;
use rayon::prelude::*;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::time::Instant;

struct Args {
    size: Option<(usize, usize)>,
    fps: Option<(u32, u32)>,
    format: Format,
    colors: usize,
    dither: Dither,
    loop_count: Option<u16>,
    lossy: u32,
    threads: Option<usize>,
    stats: bool,
}

#[derive(PartialEq)]
enum Format {
    Auto,
    Rgba,
    Y4m,
}

const USAGE: &str = "\
weft: fast parallel GIF encoder (stdin -> stdout)

usage: weft [options] < input > out.gif

input is auto-detected: yuv4mpegpipe (y4m) or raw RGBA frames.

options:
  --size WxH         frame size (required for raw RGBA input)
  --fps N[/D]        frame rate (raw RGBA default: 30; overrides y4m header)
  --format F         auto | rgba | y4m          (default: auto)
  --colors N         max palette colors, 2-256  (default: 256; one slot is
                     reserved for transparency, so 256 means 255 colors)
  --dither D         sierra2 | fs | bayer | none (default: sierra2)
  --loop N           loop count, 0 = forever    (default: 0)
  --lossy N          lossy LZW compression, 0-200 (default: 0 = lossless
                     encoding of the quantized frames; ~30 is subtle and
                     much smaller on dithered content)
  --no-loop          play once (no NETSCAPE extension)
  --threads N        worker threads             (default: all cores)
  --stats            print timing breakdown to stderr
";

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        size: None,
        fps: None,
        format: Format::Auto,
        colors: 256,
        dither: Dither::Sierra2_4a,
        loop_count: Some(0),
        lossy: 0,
        threads: None,
        stats: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--size" => {
                let v = val("--size")?;
                let (w, h) = v.split_once(['x', 'X']).ok_or("--size expects WxH")?;
                a.size = Some((
                    w.parse().map_err(|_| "bad width")?,
                    h.parse().map_err(|_| "bad height")?,
                ));
            }
            "--fps" => {
                let v = val("--fps")?;
                let (n, d) = match v.split_once(['/', ':']) {
                    Some((n, d)) => (
                        n.parse().map_err(|_| "bad fps")?,
                        d.parse().map_err(|_| "bad fps")?,
                    ),
                    None => (v.parse().map_err(|_| "bad fps")?, 1),
                };
                if n == 0 || d == 0 {
                    return Err("fps must be positive".into());
                }
                a.fps = Some((n, d));
            }
            "--format" => {
                a.format = match val("--format")?.as_str() {
                    "auto" => Format::Auto,
                    "rgba" => Format::Rgba,
                    "y4m" => Format::Y4m,
                    f => return Err(format!("unknown format {f}")),
                }
            }
            "--colors" => {
                a.colors = val("--colors")?.parse().map_err(|_| "bad --colors")?;
                if !(2..=256).contains(&a.colors) {
                    return Err("--colors must be 2-256".into());
                }
            }
            "--dither" => {
                a.dither = match val("--dither")?.as_str() {
                    "sierra2" | "sierra2_4a" => Dither::Sierra2_4a,
                    "fs" | "floyd_steinberg" => Dither::FloydSteinberg,
                    "bayer" => Dither::Bayer,
                    "none" => Dither::None,
                    d => return Err(format!("unknown dither {d}")),
                }
            }
            "--loop" => a.loop_count = Some(val("--loop")?.parse().map_err(|_| "bad --loop")?),
            "--lossy" => {
                a.lossy = val("--lossy")?.parse().map_err(|_| "bad --lossy")?;
                if a.lossy > 200 {
                    return Err("--lossy must be 0-200".into());
                }
            }
            "--no-loop" => a.loop_count = None,
            "--threads" => {
                a.threads = Some(val("--threads")?.parse().map_err(|_| "bad --threads")?)
            }
            "--stats" => a.stats = true,
            "--help" | "-h" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown option {other} (see --help)")),
        }
    }
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Some(n) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .expect("rayon pool");
    }
    if let Err(e) = run(&args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> io::Result<()> {
    let t0 = Instant::now();
    // Stdin (not StdinLock): the reader moves to a worker thread, and
    // StdinLock is not Send. Refills lock per read; at 1 MB granularity the
    // lock cost vanishes.
    let mut reader = BufReader::with_capacity(1 << 20, io::stdin());

    // ---- input detection ---------------------------------------------------
    let mut probe = [0u8; 9];
    let is_y4m = match args.format {
        Format::Y4m => true,
        Format::Rgba => false,
        Format::Auto => {
            reader.read_exact(&mut probe).map_err(|e| {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "no input on stdin")
                } else {
                    e
                }
            })?;
            &probe == b"YUV4MPEG2"
        }
    };
    let probe_consumed = args.format == Format::Auto;

    enum Source {
        Y4m(BufReader<io::Stdin>),
        Rgba(io::Chain<io::Cursor<Vec<u8>>, BufReader<io::Stdin>>),
    }

    let (meta, mut source) = if is_y4m {
        let mut line = String::new();
        io::BufRead::read_line(&mut reader, &mut line)?;
        let header = if probe_consumed {
            format!("YUV4MPEG2{line}")
        } else {
            line
        };
        if !header.starts_with("YUV4MPEG2") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a y4m stream",
            ));
        }
        let mut meta = input::parse_y4m_header(&header)?;
        if let Some((n, d)) = args.fps {
            meta.fps_num = n;
            meta.fps_den = d;
        }
        (meta, Source::Y4m(reader))
    } else {
        let (w, h) = args.size.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "--size WxH is required for raw RGBA input",
            )
        })?;
        let (n, d) = args.fps.unwrap_or((30, 1));
        let meta = VideoIn {
            width: w,
            height: h,
            fps_num: n,
            fps_den: d,
            chroma: None,
        };
        let leftover = if probe_consumed {
            probe.to_vec()
        } else {
            Vec::new()
        };
        (meta, Source::Rgba(io::Cursor::new(leftover).chain(reader)))
    };
    let (w, h) = (meta.width, meta.height);
    if w == 0 || h == 0 || w > 65535 || h > 65535 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "frame size exceeds GIF limits",
        ));
    }

    // ---- read + histogram, overlapped -------------------------------------
    // A reader thread streams frames into a bounded channel while rayon
    // workers accumulate per-thread histograms, so palette statistics are
    // (nearly) free whenever input arrives slower than it can be hashed.
    let t1 = Instant::now();
    let nthreads = rayon::current_num_threads().max(1);
    let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Frame)>(2 * nthreads);
    let (read_res, acc) = std::thread::scope(|scope| {
        let meta_ref = &meta;
        let reader_handle = scope.spawn(move || -> io::Result<usize> {
            let mut n = 0usize;
            loop {
                let frame = match &mut source {
                    Source::Y4m(r) => input::read_y4m_frame(r, meta_ref)?,
                    Source::Rgba(r) => input::read_rgba_frame(r, w, h)?,
                };
                match frame {
                    Some(f) => {
                        if tx.send((n, f)).is_err() {
                            break; // consumer died; its error surfaces below
                        }
                        n += 1;
                    }
                    None => break,
                }
            }
            Ok(n)
        });
        let acc = rx
            .into_iter()
            .par_bridge()
            .fold(
                || (palette::ColorHist::new(), Vec::new(), vec![0u8; w * 4]),
                |(mut hist, mut frames, mut row), (i, f)| {
                    let src = color::RowSource::new(&f, w, h, meta_ref.chroma);
                    for y in 0..h {
                        src.fill_row(y, &mut row);
                        palette::accumulate_frame(&mut hist, &row);
                    }
                    frames.push((i, f));
                    (hist, frames, row)
                },
            )
            .reduce_with(|(mut ha, mut fa, row), (hb, fb, _)| {
                ha.merge(&hb);
                fa.extend(fb);
                (ha, fa, row)
            });
        (reader_handle.join().expect("reader thread panicked"), acc)
    });
    let nread = read_res?;
    let Some((hist, mut indexed_frames, _)) = acc else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no frames in input",
        ));
    };
    debug_assert_eq!(indexed_frames.len(), nread);
    indexed_frames.sort_unstable_by_key(|(i, _)| *i);
    let frames: Vec<Frame> = indexed_frames.into_iter().map(|(_, f)| f).collect();
    let entries = hist.entries();
    drop(hist);
    let t_read = t0.elapsed();
    let t_hist = t1.elapsed();

    // ---- palette + nearest-color map --------------------------------------
    let t2 = Instant::now();
    let colors = palette::median_cut(&entries, args.colors - 1);
    drop(entries);
    let trans_idx = colors.len() as u8;
    let slots = colors.len() + 1;
    let gct_bits = (usize::BITS - (slots - 1).leading_zeros()).max(1) as u8;
    let min_code_size = gct_bits.max(2);
    let nearest = palette::NearestMap::build(&colors);
    let t_pal = t2.elapsed();
    if args.stats {
        eprintln!(
            "  palette: {} colors, nearest-map avg candidates/cell {:.2}",
            colors.len(),
            nearest.avg_candidates()
        );
    }

    // ---- pass 2: quantize + dither (parallel per frame) ------------------
    let t3 = Instant::now();
    let quant = Quantizer {
        colors: &colors,
        nearest: &nearest,
        trans_idx,
    };
    let results: Vec<(Vec<u8>, bool)> = frames
        .into_par_iter()
        .map_init(
            || dither::QuantScratch::new(w),
            |scratch, f| {
                let mut idx = vec![0u8; w * h];
                let src = color::RowSource::new(&f, w, h, meta.chroma);
                let has_alpha = quant.quantize(&src, w, h, args.dither, scratch, &mut idx);
                drop(f);
                (idx, has_alpha)
            },
        )
        .collect();
    let any_alpha = results.iter().any(|(_, a)| *a);
    let indexed: Vec<Vec<u8>> = results.into_iter().map(|(i, _)| i).collect();
    let t_quant = t3.elapsed();

    // ---- pass 3: delta + LZW (parallel per frame) -------------------------
    // With source alpha, transparency can't double as "unchanged" marker, so
    // fall back to full frames with restore-to-background disposal.
    let t4 = Instant::now();
    let delays = gif::frame_delays(indexed.len(), meta.fps_num, meta.fps_den);
    let disposal = if any_alpha {
        gif::DISPOSAL_BACKGROUND
    } else {
        gif::DISPOSAL_NONE
    };
    let lossy_map = (args.lossy > 0).then(|| lzw::LossyMap::build(&colors, trans_idx, args.lossy));
    let encoded: Vec<gif::EncodedFrame> = (0..indexed.len())
        .into_par_iter()
        .map_init(LzwEncoder::default, |enc, i| {
            let prev = if any_alpha || i == 0 {
                None
            } else {
                Some(indexed[i - 1].as_slice())
            };
            gif::encode_frame(
                &indexed[i],
                prev,
                w,
                h,
                trans_idx,
                min_code_size,
                delays[i],
                disposal,
                lossy_map.as_ref(),
                enc,
            )
        })
        .collect();
    let t_lzw = t4.elapsed();

    // ---- mux --------------------------------------------------------------
    let t5 = Instant::now();
    let params = gif::MuxParams {
        width: w,
        height: h,
        colors: &colors,
        trans_idx,
        gct_bits,
        loop_count: args.loop_count,
    };
    let mut out = Vec::new();
    gif::mux(&params, &encoded, &mut out);
    let mut stdout = BufWriter::with_capacity(1 << 20, io::stdout().lock());
    stdout.write_all(&out)?;
    stdout.flush()?;
    let t_mux = t5.elapsed();

    if args.stats {
        let n = encoded.len();
        eprintln!(
            "weft: {n} frames {w}x{h} @{}/{} fps, {} colors, {} bytes",
            meta.fps_num,
            meta.fps_den,
            colors.len(),
            out.len()
        );
        eprintln!(
            "  read+hist {:?} (hist span {:?})  palette+lut {:?}  quantize {:?}  delta+lzw {:?}  mux+write {:?}  total {:?}",
            t_read, t_hist, t_pal, t_quant, t_lzw, t_mux, t0.elapsed()
        );
    }
    Ok(())
}
