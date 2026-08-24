//! weft: fast parallel GIF encoder.
//! Reads raw RGBA or yuv4mpegpipe from stdin, writes GIF to stdout.

#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Make mimalloc give freed pages back to the OS promptly. The pipeline
/// frees in large phase-sized bursts (raw frames as pass 2 consumes them,
/// histogram tables after the palette, index buffers per block), and with the default
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
use input::{Frame, VideoIn};
use rayon::prelude::*;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

struct Args {
    size: Option<(usize, usize)>,
    fps: Option<(u32, u32)>,
    format: Format,
    colors: usize,
    dither: Dither,
    dither_gate: u32,
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
  --dither D         bluenoise | sierra2 | fs | bayer | none
                     (default: bluenoise, which is fast, temporally
                     stable, and compresses well; sierra2 error diffusion
                     has slightly higher visual quality but is slower and
                     shimmers frame-to-frame on animated content)
  --dither-gate N    activity gate for bluenoise, 0-720 (default: 16;
                     0 = off). Smooth regions keep full dither; busier
                     regions get progressively less, reaching none at
                     N+64 activity — texture masks palette error anyway,
                     so skipping dither there cuts noise and file size
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
        dither: Dither::BlueNoise,
        dither_gate: 16,
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
                    "bluenoise" | "bn" => Dither::BlueNoise,
                    "none" => Dither::None,
                    d => return Err(format!("unknown dither {d}")),
                }
            }
            "--dither-gate" => {
                a.dither_gate = val("--dither-gate")?
                    .parse()
                    .map_err(|_| "bad --dither-gate")?;
                // 720 > the maximum activity a pixel can have minus the
                // ramp, i.e. everything above this means "never gate off"
                if a.dither_gate > 720 {
                    return Err("--dither-gate must be 0-720".into());
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
    // A reader thread streams frames into a bounded channel; the main
    // thread drains it in batches and runs each batch through two parallel
    // phases. (A) Every frame is converted to RGB-key runs and its runs are
    // counting-sorted into 256 buckets by red byte. (B) Each bucket is
    // summed into that bucket's own histogram by one task. Partitioning by
    // color instead of by frame means each color is hashed once, into one
    // small (mostly L2-resident) table; there is no per-worker duplicate
    // state to merge or dedup afterwards — bucket order is sorted order —
    // and pass 1 can use every thread instead of a capped pool. (The
    // previous frame-partitioned design had every worker growing a table
    // holding most of the clip's colors, then a serial multi-million-entry
    // sort+dedup to merge them.) With a single thread there is nothing to
    // parallelize, so the runs are added straight from each row's RLE
    // buffer instead of being materialized per frame. Alpha presence is
    // detected here too: pass 2+3 needs it before the first frame is
    // quantized.
    let t1 = Instant::now();
    let nthreads = rayon::current_num_threads().max(1);
    // A batch is whatever the reader has queued when the previous batch
    // finishes — small on a slow source (maximum overlap with input), the
    // cap on a fast one. The cap bounds the routed runs, which can reach 8
    // bytes per pixel on noisy content, to a modest transient.
    let batch_cap = 2 * nthreads;
    let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Frame)>(batch_cap);
    const BUCKETS: usize = 256;
    // Bucket state: one exact table per red-byte bucket, until the tables
    // together exceed GRID_SIZE distinct colors. Past that the palette
    // input is getting grid-folded regardless (see maybe_fold), so the
    // tables fold into 6-bit bins — one bin array per red slab (r >> 2,
    // four adjacent buckets; the 64 slabs together are exactly the 8 MB
    // fold grid, and `bins` is empty until the switch) — and every later
    // add is an indexed sum. Bin sums are commutative integers and folding
    // exact entries yields the same sums as binning the pixels directly,
    // so the result is identical whenever the switch happens, and
    // identical to folding the full exact histogram. Every bucket switches
    // at the same batch boundary, so one flag covers all of them and the
    // per-run hot loops branch on nothing but the bucket index.
    let mut hists: Vec<palette::ColorHist> = (0..BUCKETS)
        .map(|_| palette::ColorHist::with_capacity(1 << 10))
        .collect();
    let mut bins: Vec<Vec<[u64; 4]>> = Vec::new();
    let fold_slabs = |hists: &[palette::ColorHist]| -> Vec<Vec<[u64; 4]>> {
        hists
            .par_chunks(4)
            .map(|slab| {
                let mut b = vec![[0u64; 4]; palette::SLAB_BINS];
                for h in slab {
                    palette::accumulate_runs_coarse(&mut b, &h.entries());
                }
                b
            })
            .collect()
    };
    struct Routed {
        idx: usize,
        frame: Frame,
        alpha: bool,
        runs: Vec<(u32, u32)>, // this frame's runs, bucket-sorted
        offs: Vec<u32>,        // BUCKETS + 1 bucket boundaries into runs
    }
    let mut coarse = false;
    // Per-frame run buffers are recycled between batches: a fresh ~MB
    // allocation per frame (an mmap plus a page fault per 4 KiB) costs
    // more than the hashing itself at low thread counts.
    let run_pool: Mutex<Vec<Vec<(u32, u32)>>> = Mutex::new(Vec::new());
    let (read_res, mut indexed_frames, any_alpha) = std::thread::scope(|scope| {
        let meta_ref = &meta;
        let reader_handle = scope.spawn(move || -> io::Result<usize> {
            // On a fast source (tmpfs, a pipe from a decoder already
            // ahead of us) most of the reader's time is page-faulting the
            // fresh buffer for each frame, not copying into it. A helper
            // thread allocates and first-touches the buffers a few frames
            // ahead so the reader only does the read.
            let fsize = match &source {
                Source::Y4m(_) => meta_ref.chroma.unwrap().frame_bytes(w, h),
                Source::Rgba(_) => w * h * 4,
            };
            let (btx, brx) = std::sync::mpsc::sync_channel::<Vec<u8>>(4);
            let prefault = std::thread::spawn(move || loop {
                // Deliberately not `vec![0; fsize]`: that is calloc, whose
                // pages stay unmapped until first written, i.e. the fault
                // would land on the reader after all. resize() stores the
                // zeros itself, which is the first touch we want here.
                #[allow(clippy::slow_vector_initialization)]
                let v: Vec<u8> = {
                    let mut v = Vec::with_capacity(fsize);
                    v.resize(fsize, 0);
                    v
                };
                if btx.send(v).is_err() {
                    break;
                }
            });
            let mut n = 0usize;
            let res = (|| {
                loop {
                    let buf = brx.recv().expect("prefault thread died");
                    let frame = match &mut source {
                        Source::Y4m(r) => input::read_y4m_frame(r, buf)?,
                        Source::Rgba(r) => input::read_rgba_frame(r, buf)?,
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
            })();
            drop(brx); // unblocks the helper's pending send
            prefault.join().expect("prefault thread panicked");
            res
        });
        let mut frames: Vec<(usize, Frame)> = Vec::new();
        let mut any_alpha = false;
        let (mut row1, mut keys1) = (vec![0u8; w * 4], vec![0u32; w]);
        let mut runs1: Vec<(u32, u32)> = Vec::new();
        loop {
            let mut batch: Vec<(usize, Frame)> = Vec::with_capacity(batch_cap);
            match rx.recv() {
                Ok(f) => batch.push(f),
                Err(_) => break, // reader done and channel drained
            }
            while batch.len() < batch_cap {
                match rx.try_recv() {
                    Ok(f) => batch.push(f),
                    Err(_) => break,
                }
            }
            if nthreads == 1 {
                // Nothing to parallelize: skip materializing the runs and
                // add them straight into the bucket tables (on dithered
                // content the runs are most of a byte per pixel per pass,
                // and one thread pays that traffic in full).
                if coarse && bins.is_empty() {
                    bins = fold_slabs(&hists);
                    hists = Vec::new();
                }
                // per row: RLE into an L1-resident run buffer, then add
                // with the table slot a few runs ahead prefetched — the
                // bucket tables together are still a few MB, and each add
                // otherwise serializes behind its miss
                let add_runs = |hists: &mut [palette::ColorHist],
                                bins: &mut [Vec<[u64; 4]>],
                                runs: &[(u32, u32)]| {
                    if coarse {
                        for &(c, n) in runs {
                            palette::add_run_coarse(&mut bins[(c >> 18) as usize], c, n);
                        }
                    } else {
                        for j in 0..runs.len() {
                            if let Some(&(c, _)) = runs.get(j + 8) {
                                hists[(c >> 16) as usize].prefetch(c);
                            }
                            let (c, n) = runs[j];
                            hists[(c >> 16) as usize].add(c, n);
                        }
                    }
                };
                for (i, f) in batch {
                    let mut alpha = false;
                    {
                        let src = color::RowSource::new(&f, w, h, meta_ref.chroma);
                        if src.has_direct_rgb_keys() {
                            for y in 0..h {
                                src.fill_rgb_keys(y, &mut keys1);
                                palette::scan_rgb_key_runs(&keys1, &mut runs1);
                                add_runs(&mut hists, &mut bins, &runs1);
                            }
                        } else {
                            for y in 0..h {
                                let rgba = rgba_row(&src, y, &mut row1);
                                alpha |= palette::scan_rgba_runs(rgba, &mut runs1);
                                add_runs(&mut hists, &mut bins, &runs1);
                            }
                        }
                    }
                    let frame = match f {
                        Frame::Rgba(rgba) if !alpha => Frame::Rgb(color::rgba_to_rgb(&rgba)),
                        other => other,
                    };
                    any_alpha |= alpha;
                    frames.push((i, frame));
                }
                if !coarse {
                    let distinct: usize = hists.iter().map(|h| h.len()).sum();
                    coarse = distinct > palette::GRID_SIZE;
                }
                continue;
            }
            // phase A: frames -> bucket-sorted runs, all threads
            let routed: Vec<Routed> = batch
                .into_par_iter()
                .map_init(
                    || (vec![0u8; w * 4], vec![0u32; w], Vec::new()),
                    |(row, keys, all), (i, f)| {
                        all.clear();
                        let mut counts = [0u32; BUCKETS];
                        let mut alpha = false;
                        {
                            let src = color::RowSource::new(&f, w, h, meta_ref.chroma);
                            if src.has_direct_rgb_keys() {
                                for y in 0..h {
                                    src.fill_rgb_keys(y, keys);
                                    palette::scan_rgb_key_runs_counted(keys, all, &mut counts);
                                }
                            } else {
                                for y in 0..h {
                                    alpha |= palette::scan_rgba_runs_counted(
                                        rgba_row(&src, y, row),
                                        all,
                                        &mut counts,
                                    );
                                }
                            }
                        }
                        let mut runs = run_pool.lock().unwrap().pop().unwrap_or_default();
                        let offs = palette::bucket_runs(all, &counts, &mut runs);
                        // The scan just told us whether this frame uses any
                        // transparency; when it doesn't, the alpha plane is
                        // a constant and the frame can be packed to RGB for
                        // the rest of its (clip-long) life. The RGBA buffer
                        // is freed immediately, so the extra resident bytes
                        // are one frame per busy worker, not one per clip.
                        let frame = match f {
                            Frame::Rgba(rgba) if !alpha => Frame::Rgb(color::rgba_to_rgb(&rgba)),
                            other => other,
                        };
                        Routed { idx: i, frame, alpha, runs, offs }
                    },
                )
                .collect();
            // phase B: one task per bucket, all of the batch's runs for it
            if coarse && bins.is_empty() {
                bins = fold_slabs(&hists);
                hists = Vec::new();
            }
            if coarse {
                bins.par_iter_mut().enumerate().for_each(|(g, slab)| {
                    for r in &routed {
                        let s = &r.runs[r.offs[4 * g] as usize..r.offs[4 * g + 4] as usize];
                        palette::accumulate_runs_coarse(slab, s);
                    }
                });
            } else {
                hists.par_iter_mut().enumerate().for_each(|(b, hist)| {
                    for r in &routed {
                        let s = &r.runs[r.offs[b] as usize..r.offs[b + 1] as usize];
                        palette::accumulate_runs(hist, s);
                    }
                });
                let distinct: usize = hists.iter().map(|h| h.len()).sum();
                coarse = distinct > palette::GRID_SIZE;
            }
            let mut pool = run_pool.lock().unwrap();
            for r in routed {
                any_alpha |= r.alpha;
                frames.push((r.idx, r.frame));
                pool.push(r.runs);
            }
        }
        (
            reader_handle.join().expect("reader thread panicked"),
            frames,
            any_alpha,
        )
    });
    let nread = read_res?;
    if indexed_frames.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no frames in input",
        ));
    }
    debug_assert_eq!(indexed_frames.len(), nread);
    indexed_frames.sort_unstable_by_key(|(i, _)| *i);
    let frames: Vec<Frame> = indexed_frames.into_iter().map(|(_, f)| f).collect();
    let coarse_binned = coarse;
    let entries: Vec<(u32, u32)> = if coarse {
        if bins.is_empty() {
            // the switch came at the last batch: nothing has folded yet
            bins = fold_slabs(&hists);
            hists = Vec::new();
        }
        // slab order is grid order, and each slab's bins are in grid order
        let slabs: Vec<Vec<(u32, u32)>> = bins
            .par_iter()
            .map(|b| palette::fold_bins_to_entries(b))
            .collect();
        slabs.concat()
    } else {
        // Each color lives in exactly one bucket, so per-bucket sorted
        // entries concatenate to the sorted, deduplicated histogram
        // (median_cut's canonicalizing sort is then a no-op).
        let per: Vec<Vec<(u32, u32)>> = hists
            .par_iter()
            .map(|h| {
                let mut e = h.entries();
                e.sort_unstable();
                e
            })
            .collect();
        per.concat()
    };
    drop(hists);
    drop(bins);
    let t_read = t0.elapsed();
    let t_hist = t1.elapsed();

    // ---- palette + nearest-color map --------------------------------------
    let t2 = Instant::now();
    let n_entries = entries.len();
    let entries = palette::maybe_fold(entries);
    let n_folded = entries.len();
    let colors = palette::median_cut(entries, args.colors - 1);
    let t_mc = t2.elapsed();
    let trans_idx = colors.len() as u8;
    let slots = colors.len() + 1;
    let gct_bits = (usize::BITS - (slots - 1).leading_zeros()).max(1) as u8;
    let min_code_size = gct_bits.max(2);
    let nearest = palette::NearestMap::build(&colors);
    let t_pal = t2.elapsed();
    if args.stats {
        let folded = if coarse_binned {
            " (coarse-binned)".into()
        } else if n_folded != n_entries {
            format!(" (folded to {n_folded})")
        } else {
            String::new()
        };
        eprintln!(
            "  palette: {} colors from {} entries{}, nearest-map avg candidates/cell {:.2}, median_cut {:?}",
            colors.len(),
            n_entries,
            folded,
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
        gate: args.dither_gate,
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
    // Index buffers persist across blocks (every quantize mode overwrites
    // all w*h bytes, so recycling needs no clearing): a clip-length run
    // faults each buffer's pages once instead of allocating, zeroing, and
    // freeing ~w*h bytes per frame.
    let mut idx_block: Vec<Vec<u8>> = Vec::new();
    // `for_each_init`/`map_init` state lives for only one parallel
    // operation. Since the block loop launches two new operations per
    // block, using them here would repeatedly allocate and zero the 512 KiB
    // nearest-color cache and recreate the encoder buffers. Keep one state
    // bundle per Rayon worker for the whole clip instead. Initialize each
    // half lazily on the worker that first uses it: eagerly constructing
    // nthreads caches here serializes tens of MiB of first-touch writes on
    // large machines, and an encode-only worker never needs QuantScratch.
    // A worker executes only one closure at a time, so these locks are
    // uncontended; they provide safe indexed ownership across independent
    // parallel calls.
    struct WorkerCtx {
        quant: OnceLock<Mutex<dither::QuantScratch>>,
        encode: OnceLock<Mutex<gif::EncodeCtx>>,
    }
    // The extra slot covers Rayon's single-item/sequential fast path, which
    // can execute the closure on the calling thread (and therefore has no
    // Rayon worker index).
    let worker_ctx: Vec<WorkerCtx> = (0..=nthreads)
        .map(|_| WorkerCtx {
            quant: OnceLock::new(),
            encode: OnceLock::new(),
        })
        .collect();
    while start < nread {
        let chunk: Vec<Frame> = frames_it.by_ref().take(block).collect();
        let cn = chunk.len();
        if idx_block.len() < cn {
            // Allocate (and first-touch) the block's index buffers in
            // parallel: done serially this is several ms of page faults
            // per clip on a wide machine, all before any worker starts.
            let extra: Vec<Vec<u8>> = (idx_block.len()..cn)
                .into_par_iter()
                .map(|_| vec![0u8; w * h])
                .collect();
            idx_block.extend(extra);
        }
        chunk
            .into_par_iter()
            .zip(idx_block[..cn].par_iter_mut())
            .for_each(|(f, idx)| {
                let wi = rayon::current_thread_index().unwrap_or(nthreads);
                let mut scratch = worker_ctx[wi]
                    .quant
                    .get_or_init(|| Mutex::new(dither::QuantScratch::new(w)))
                    .lock()
                    .unwrap();
                let src = color::RowSource::new(&f, w, h, meta.chroma);
                quant.quantize(&src, w, h, args.dither, &mut scratch, idx);
            });
        encoded.par_extend((0..cn).into_par_iter().map(|j| {
            let wi = rayon::current_thread_index().unwrap_or(nthreads);
            let mut encode = worker_ctx[wi]
                .encode
                .get_or_init(|| Mutex::new(gif::EncodeCtx::default()))
                .lock()
                .unwrap();
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
                &mut encode,
            )
        }));
        // The block's last indexed frame seeds the next block's first
        // delta; swap keeps both buffers in the recycled pool.
        match prev_last.as_mut() {
            Some(pl) => std::mem::swap(pl, &mut idx_block[cn - 1]),
            None => prev_last = Some(std::mem::replace(&mut idx_block[cn - 1], vec![0u8; w * h])),
        }
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

/// Row `y` of `src` as RGBA: borrowed straight from the frame when it is
/// already stored that way, otherwise converted into `scratch` (len w*4).
/// RGBA input is by far the hot case, and copying every source byte into a
/// scratch row just to read it back is pure overhead.
#[inline]
fn rgba_row<'r>(src: &color::RowSource<'r>, y: usize, scratch: &'r mut [u8]) -> &'r [u8] {
    match src.rgba_row(y) {
        Some(borrowed) => borrowed,
        None => {
            src.fill_row(y, scratch);
            scratch
        }
    }
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
