//! weft: fast parallel GIF encoder.
//! Reads raw RGBA or yuv4mpegpipe from stdin, writes GIF to stdout.

#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Make mimalloc give freed pages back to the OS promptly. The pipeline
/// frees in large phase-sized bursts (raw frames after packing, histogram
/// tables after the palette, index buffers per block), and with the default
/// 10ms purge delay + no abandoned-page purging those bursts linger and the
/// static binary peaks 20-30% above the glibc build. A 1ms delay recovers
/// nearly all of that while still letting the reader's hot frame buffers
/// recycle without madvise churn (0 costs real throughput on fast inputs);
/// abandoned-page purging matters because the reader thread — which
/// allocates every raw frame — exits after pass 1, and its heap's pages
/// otherwise stay resident until another thread happens to reclaim them.
#[cfg(target_env = "musl")]
fn tune_mimalloc() {
    // Indices into mimalloc's mi_option_t enum; libmimalloc-sys names only
    // a subset of the options, but the layout is fixed by the mimalloc
    // release the locked libmimalloc-sys bundles.
    const MI_OPTION_ABANDONED_PAGE_PURGE: i32 = 12;
    const MI_OPTION_PURGE_DELAY: i32 = 15; // milliseconds
    unsafe {
        libmimalloc_sys::mi_option_set(MI_OPTION_ABANDONED_PAGE_PURGE, 1);
        libmimalloc_sys::mi_option_set(MI_OPTION_PURGE_DELAY, 1);
    }
}

#[cfg(not(target_env = "musl"))]
fn tune_mimalloc() {}

mod bluenoise;
mod color;
mod dither;
mod gif;
mod input;
mod lzw;
mod oklab;
mod palette;
mod simdops;

use dither::{Dither, Quantizer};
use input::{Frame, StoredFrame, VideoIn};
use lzw::LzwEncoder;
use rayon::prelude::*;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::sync::Mutex;
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
    compress: bool,
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
  --dither D         bluenoise | sierra2 | fs | bayer | none
                     (default: bluenoise, which is fast, temporally
                     stable, and compresses well; sierra2 error diffusion
                     has slightly higher visual quality but is slower and
                     shimmers frame-to-frame on animated content)
  --loop N           loop count, 0 = forever    (default: 0)
  --lossy N          lossy LZW compression, 0-200 (default: 0 = lossless
                     encoding of the quantized frames; ~30 is subtle and
                     much smaller on dithered content)
  --no-loop          play once (no NETSCAPE extension)
  --no-compress      buffer frames raw in memory instead of LZ4-packing
                     them between passes (output is identical; raises
                     peak memory, mainly for benchmarking the tradeoff)
  --threads N        worker threads             (default: all cores)
  --stats            print timing breakdown to stderr
";

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        size: None,
        fps: None,
        format: Format::Auto,
        colors: 256,
        dither: Dither::BlueNoise,
        loop_count: Some(0),
        lossy: 0,
        threads: None,
        stats: false,
        compress: true,
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
                    "bluenoise" | "bn" => Dither::BlueNoise,
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
            "--no-compress" => a.compress = false,
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
    tune_mimalloc();
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
    // Each frame is LZ4-packed by the worker that histogrammed it (while
    // its bytes are still cache-warm), so the set buffered for pass 2
    // shrinks to roughly the compressed size of the input. Alpha presence
    // is detected here too — pass 2+3 needs it before the first frame is
    // quantized.
    let t1 = Instant::now();
    let nthreads = rayon::current_num_threads().max(1);
    let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Frame)>(2 * nthreads);
    // Worker-local histograms flush to a shared list of entry runs once
    // they exceed this many distinct colors. This bounds each worker's
    // open-addressed table to a few MB — on true-color content the tables
    // otherwise balloon to hundreds of MB per thread — while keeping the
    // lock trivially cheap: a flush converts the table to a plain entry
    // vector (linear scan, done outside any lock) and the critical section
    // is a single Vec push. Colors counted by several runs are summed by
    // one parallel sort + adjacent-merge afterwards.
    const HIST_SPILL: usize = 1 << 20;
    let spilled: Mutex<Vec<Vec<(u32, u32)>>> = Mutex::new(Vec::new());
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
                || {
                    (
                        palette::ColorHist::new(),
                        Vec::new(),
                        vec![0u8; w * 4],
                        false,
                    )
                },
                |(mut hist, mut frames, mut row, mut alpha), (i, f)| {
                    let src = color::RowSource::new(&f, w, h, meta_ref.chroma);
                    for y in 0..h {
                        src.fill_row(y, &mut row);
                        alpha |= palette::accumulate_frame(&mut hist, &row);
                    }
                    if hist.len() > HIST_SPILL {
                        let run = hist.entries();
                        spilled.lock().unwrap().push(run);
                        hist = palette::ColorHist::new();
                    }
                    frames.push((i, StoredFrame::pack(f, args.compress)));
                    (hist, frames, row, alpha)
                },
            )
            .reduce_with(|(ha, mut fa, row, aa), (hb, fb, _, ab)| {
                // Flush instead of hash-merging: reductions run while other
                // workers still accumulate, and a table-into-table merge of
                // millions of colors would serialize them behind cache-miss
                // heavy rehashing. The sort below dedups across runs.
                let run = hb.entries();
                if !run.is_empty() {
                    spilled.lock().unwrap().push(run);
                }
                fa.extend(fb);
                (ha, fa, row, aa | ab)
            });
        (reader_handle.join().expect("reader thread panicked"), acc)
    });
    let nread = read_res?;
    let Some((hist, mut indexed_frames, _, any_alpha)) = acc else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no frames in input",
        ));
    };
    debug_assert_eq!(indexed_frames.len(), nread);
    indexed_frames.sort_unstable_by_key(|(i, _)| *i);
    let frames: Vec<StoredFrame> = indexed_frames.into_iter().map(|(_, f)| f).collect();
    // Concatenate all flushed runs plus the surviving accumulator, then sum
    // duplicate colors with one parallel sort + adjacent merge. median_cut
    // canonicalizes its input by the same sort, so this costs nothing extra
    // beyond the (already tight) concatenated allocation.
    let mut runs = spilled.into_inner().unwrap();
    runs.push(hist.entries());
    drop(hist);
    let mut entries = Vec::with_capacity(runs.iter().map(Vec::len).sum());
    while let Some(r) = runs.pop() {
        entries.extend_from_slice(&r); // pop + drop each run: concat never
                                       // holds two full copies
    }
    drop(runs);
    if entries.len() > 16384 {
        entries.par_sort_unstable();
    } else {
        entries.sort_unstable();
    }
    entries.dedup_by(|cur, prev| {
        if cur.0 == prev.0 {
            prev.1 = prev.1.saturating_add(cur.1);
            true
        } else {
            false
        }
    });
    entries.shrink_to_fit(); // drops the duplicates' share before the
                             // palette build allocates its scratch
    let t_read = t0.elapsed();
    let t_hist = t1.elapsed();

    // ---- palette + nearest-color map --------------------------------------
    let t2 = Instant::now();
    let n_entries = entries.len();
    let colors = palette::median_cut(&entries, args.colors - 1);
    let t_mc = t2.elapsed();
    drop(entries);
    let trans_idx = colors.len() as u8;
    let slots = colors.len() + 1;
    let gct_bits = (usize::BITS - (slots - 1).leading_zeros()).max(1) as u8;
    let min_code_size = gct_bits.max(2);
    let nearest = palette::NearestMap::build(&colors);
    let t_pal = t2.elapsed();
    if args.stats {
        eprintln!(
            "  palette: {} colors from {} entries, nearest-map avg candidates/cell {:.2}, median_cut {:?}",
            colors.len(),
            n_entries,
            nearest.avg_candidates(),
            t_mc
        );
    }

    // ---- pass 2+3: quantize + dither + delta + LZW, fused in blocks -------
    // Fusing keeps the full indexed set (frames x w*h) from ever being
    // resident: frames are processed in blocks — every frame in a block
    // quantizes in parallel, then every frame encodes against its
    // predecessor's indices in parallel (quantization is deterministic, and
    // with disposal "none" the canvas after frame i-1 is exactly frame i-1's
    // indices), and only the block's last indexed frame survives to seed the
    // next block's first delta. With source alpha, transparency can't double
    // as "unchanged" marker, so frames encode whole with
    // restore-to-background disposal (any_alpha comes from pass 1).
    let t3 = Instant::now();
    let quant = Quantizer {
        nearest: &nearest,
        trans_idx,
        // median_cut returns the exact colors when they all fit
        exact_palette: n_entries < args.colors,
    };
    let delays = gif::frame_delays(nread, meta.fps_num, meta.fps_den);
    let disposal = if any_alpha {
        gif::DISPOSAL_BACKGROUND
    } else {
        gif::DISPOSAL_NONE
    };
    let lossy_map = (args.lossy > 0).then(|| lzw::LossyMap::build(&colors, trans_idx, args.lossy));
    // Enough frames per block that both halves keep every worker busy;
    // small enough that the block's index buffers stay a modest, clip-
    // length-independent working set.
    let block = 4 * nthreads;
    let mut encoded: Vec<gif::EncodedFrame> = Vec::with_capacity(nread);
    let mut prev_last: Option<Vec<u8>> = None;
    let mut frames_it = frames.into_iter();
    let mut start = 0usize;
    while start < nread {
        let chunk: Vec<StoredFrame> = frames_it.by_ref().take(block).collect();
        let cn = chunk.len();
        let idx_block: Vec<Vec<u8>> = chunk
            .into_par_iter()
            .map_init(
                || (dither::QuantScratch::new(w), Vec::new()),
                |(scratch, raw), f| {
                    let mut idx = vec![0u8; w * h];
                    let src = color::RowSource::from_bytes(f.unpack(raw), w, h, meta.chroma);
                    quant.quantize(&src, w, h, args.dither, scratch, &mut idx);
                    idx
                },
            )
            .collect();
        encoded.par_extend(
            (0..cn)
                .into_par_iter()
                .map_init(LzwEncoder::default, |enc, j| {
                    let i = start + j;
                    let prev = if any_alpha || i == 0 {
                        None
                    } else if j == 0 {
                        prev_last.as_deref()
                    } else {
                        Some(idx_block[j - 1].as_slice())
                    };
                    gif::encode_frame(
                        &idx_block[j],
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
                }),
        );
        prev_last = idx_block.into_iter().next_back();
        start += cn;
    }
    let t_qlzw = t3.elapsed();

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
    let mut stdout = CountWriter {
        inner: BufWriter::with_capacity(1 << 20, io::stdout().lock()),
        written: 0,
    };
    gif::mux(&params, &encoded, &mut stdout)?;
    stdout.flush()?;
    let t_mux = t5.elapsed();

    if args.stats {
        let n = encoded.len();
        eprintln!(
            "weft: {n} frames {w}x{h} @{}/{} fps, {} colors, {} bytes",
            meta.fps_num,
            meta.fps_den,
            colors.len(),
            stdout.written
        );
        eprintln!(
            "  read+hist {:?} (hist span {:?})  palette+lut {:?}  quantize+lzw {:?}  mux+write {:?}  total {:?}",
            t_read, t_hist, t_pal, t_qlzw, t_mux, t0.elapsed()
        );
    }
    Ok(())
}

/// Passthrough writer that counts bytes for the --stats report.
struct CountWriter<W> {
    inner: W,
    written: u64,
}

impl<W: Write> Write for CountWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
