//! Per-frame quantization to palette indices, with optional error-diffusion
//! or ordered dithering. Frames are independent, so this stage parallelizes
//! across frames; error diffusion stays serial only *within* a frame.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dither {
    /// Sierra-2-4A ("filter lite"): ffmpeg paletteuse's default.
    Sierra2_4a,
    /// 8x8 ordered Bayer: a fixed threshold matrix, so it compresses
    /// better and is more temporally stable than blue noise, at the cost
    /// of visible cross-hatch structure.
    Bayer,
    /// 64x64 void-and-cluster blue-noise two-candidate ordered dither: no
    /// serial error-diffusion chain, temporally stable, far less visible
    /// structure than Bayer.
    BlueNoise,
    /// Blue noise gated per 32x32 tile by a banding detector: tiles whose
    /// nearest-colour map shows contours between long runs of visibly
    /// different colours (the signature of posterization) are dithered,
    /// everything else takes plain nearest colour. See `BandGate`.
    Auto,
    None,
}

/// Banding detector for `Dither::Auto`. A band contour is a boundary
/// between two same-index runs of at least `RUN` pixels whose two palette
/// colours differ by an OkLab distance that is visible but not an edge
/// (`pairs`, built by `NearestMap::band_pair_table`). Grain blobs fail the
/// run test, outlines fail the distance test, gradients pass both. A tile
/// is dithered when more than 1/`DENSITY` of its pixels sit on such a
/// contour. Measured on a 32x32 grid: mandelbrot gradients 31% of tiles,
/// dark live-action panelling (visibly posterized undithered) 19%, flat
/// cel animation 4%.
pub struct BandGate {
    pub pairs: Vec<u8>,
    /// live / total tile counts across the run, for --stats
    pub live_tiles: std::sync::atomic::AtomicU64,
    pub total_tiles: std::sync::atomic::AtomicU64,
}

impl BandGate {
    pub const RUN: usize = 8;
    pub const TILE: usize = 32;
    pub const DENSITY: u32 = 50; // 2%
    pub const DE_LO: f32 = 0.012;
    pub const DE_HI: f32 = 0.05;
    /// Live-tile fraction above which the whole frame is dithered: 1/4.
    pub const FULL_NUM: u64 = 1;
    pub const FULL_DENOM: u64 = 4;

    pub fn new(nearest: &crate::palette::NearestMap) -> Self {
        BandGate {
            pairs: nearest.band_pair_table(Self::DE_LO, Self::DE_HI),
            live_tiles: std::sync::atomic::AtomicU64::new(0),
            total_tiles: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

const BAYER8: [[u8; 8]; 8] = [
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
];

pub struct Quantizer<'a> {
    pub nearest: &'a crate::palette::NearestMap,
    pub trans_idx: u8,
    /// The palette reproduces every source color exactly (fewer distinct
    /// colors than palette slots): every c1 lookup has zero error, so the
    /// blue-noise mode's per-pixel early-out beats the staged pipeline.
    pub exact_palette: bool,
    /// Activity gate for the blue-noise mode: dither at full strength
    /// where local activity (see `simdops::att_scalar`) is at or below
    /// this, ramp to none over the next 64 units. Smooth gradients stay
    /// dithered (that's what hides banding); texture and edges — where
    /// palette error is masked by the content and dither reads as pure
    /// noise — degrade to plain nearest-color, which also compresses far
    /// better. 0 disables the gate (dither everywhere error is nonzero).
    pub gate: u32,
    /// Banding detector for `Dither::Auto` (None: mode unavailable).
    pub band: Option<&'a BandGate>,
}

/// Per-thread quantization scratch: RGBA row buffers, memo cache, and the
/// per-pixel arrays of the staged blue-noise pipeline.
pub struct QuantScratch {
    pub cache: crate::palette::IdxCache,
    row: Vec<u8>,
    row2: Vec<u8>,
    keys: Vec<u32>,
    keys2: Vec<u32>,
    pk1: Vec<u32>,
    pk2: Vec<u32>,
    ors: Vec<u32>,
    c2c: Vec<u32>,
    att: Vec<u32>,
    // Dither::Auto: whole-frame nearest indices, the previous row's
    // long-run flags, and per-tile contour counts / live flags
    idx_frame: Vec<u8>,
    // padded rows for the SIMD scorer (BAND_PAD each side)
    idxp: Vec<u8>,
    flat_prev: Vec<u8>,
    flat_cur: Vec<u8>,
    ltmp: Vec<u8>,
    cand: Vec<u8>,
    tile_counts: Vec<u32>,
    tile_live: Vec<u8>,
}

impl QuantScratch {
    pub fn new(w: usize) -> Self {
        QuantScratch {
            cache: crate::palette::IdxCache::default(),
            row: vec![0u8; w * 4],
            row2: vec![0u8; w * 4],
            keys: vec![0; w],
            keys2: vec![0; w],
            pk1: vec![0; w],
            pk2: vec![0; w],
            ors: vec![0; w],
            c2c: vec![0; w],
            // 256 = no attenuation: with the gate off this is never
            // rewritten and the threshold pick reduces to the ungated one
            att: vec![256; w],
            idx_frame: Vec::new(),
            idxp: vec![0xFF; w + 2 * crate::simdops::BAND_PAD],
            flat_prev: vec![0; w + 2 * crate::simdops::BAND_PAD],
            flat_cur: vec![0; w + 2 * crate::simdops::BAND_PAD],
            ltmp: vec![0; w + 2 * crate::simdops::BAND_PAD],
            cand: vec![0; w],
            tile_counts: Vec::new(),
            tile_live: Vec::new(),
        }
    }
}

impl<'a> Quantizer<'a> {
    /// Quantize a frame (accessed row-by-row via `src`, so YUV conversion
    /// stays fused and cache-resident) into palette indices. Returns true
    /// if any pixel was alpha-transparent.
    pub fn quantize(
        &self,
        src: &crate::color::RowSource,
        w: usize,
        h: usize,
        mode: Dither,
        scratch: &mut QuantScratch,
        out: &mut [u8],
    ) -> bool {
        match mode {
            Dither::None => self.quantize_plain(src, w, h, scratch, out),
            Dither::Bayer => self.quantize_bayer(src, w, h, scratch, out),
            Dither::BlueNoise => self.quantize_bluenoise(src, w, h, scratch, out),
            // an exact palette has nothing to dither; without a detector
            // fall back to plain blue noise
            Dither::Auto if self.exact_palette => self.quantize_plain(src, w, h, scratch, out),
            Dither::Auto => match self.band {
                Some(b) => self.quantize_bluenoise_auto(src, w, h, scratch, out, b),
                None => self.quantize_bluenoise(src, w, h, scratch, out),
            },
            Dither::Sierra2_4a => self.quantize_diffuse(src, w, h, scratch, out),
        }
    }

    /// Two-candidate blue-noise ordered dither: c1 = nearest(p); if
    /// quantization error is nonzero, c2 = nearest across the error
    /// direction, and the threshold picks between c1 and c2 in proportion
    /// to where p sits between them, scaled by the activity gate's
    /// per-pixel attenuation (`gate` > 0). Pixels with an exact palette
    /// match never dither, so flat regions stay clean and delta-friendly;
    /// gated pixels degrade to plain nearest-color.
    ///
    /// There is no cross-pixel dependency, so each row runs as staged
    /// passes: an attenuation pass against the previous row (gate on),
    /// SIMD key computation, a prefetched memo-cache-first lookup pass,
    /// SIMD probe-color math, the same lookup pass for the far candidates
    /// (skipping exact and fully attenuated pixels), and a SIMD threshold
    /// pick. Results are identical to the per-pixel formulation — the
    /// lookups and integer math are the same, just reordered. Tiles that
    /// can't dither at all — every pixel exact, or every pixel fully
    /// attenuated — skip the far-candidate stages and emit c1 directly.
    // the lookup loops index several parallel stage arrays at fixed
    // lookahead offsets; an iterator form would obscure that
    #[allow(clippy::needless_range_loop)]
    fn quantize_bluenoise(
        &self,
        src: &crate::color::RowSource,
        w: usize,
        h: usize,
        scratch: &mut QuantScratch,
        out: &mut [u8],
    ) -> bool {
        if self.exact_palette {
            return self.quantize_bluenoise_scalar(src, w, h, scratch, out);
        }
        // Tile width: multiple of 64 so the blue-noise mask stays aligned
        // to tile starts, small enough that a tile's stage arrays (~28
        // bytes per pixel) stay L1-resident between passes.
        const TILE: usize = 256;
        let mask32 = &crate::bluenoise::BLUE_NOISE_64_U32;
        let level = crate::simdops::level();
        let gate_on = self.gate > 0;
        let mut has_alpha = false;
        for y in 0..h {
            let keys_ready = src.fill_row_with_grid_keys(y, &mut scratch.row, &mut scratch.keys);
            // stage 1 runs row-wide: grid keys + alpha presence, with the
            // activity attenuation fused into the same pass over the
            // pixels when the gate is on (row 0 has no upper neighbor, so
            // it gates on horizontal activity alone)
            let has_alpha_row = if gate_on && keys_ready {
                let prev: &[u8] = if y == 0 { &scratch.row } else { &scratch.row2 };
                crate::simdops::bn_activity(level, &scratch.row, prev, self.gate, &mut scratch.att);
                false
            } else if gate_on {
                let prev: &[u8] = if y == 0 { &scratch.row } else { &scratch.row2 };
                crate::simdops::bn_keys_att(
                    level,
                    &scratch.row,
                    prev,
                    self.gate,
                    &mut scratch.keys,
                    &mut scratch.att,
                )
            } else if !keys_ready {
                crate::simdops::bn_keys(level, &scratch.row, &mut scratch.keys)
            } else {
                false
            };
            has_alpha |= has_alpha_row;
            let mrow: &[u32; 64] = mask32[(y & 63) << 6..((y & 63) << 6) + 64]
                .try_into()
                .unwrap();
            let mut x0 = 0usize;
            while x0 < w {
                let tw = TILE.min(w - x0);
                let row = &scratch.row[x0 * 4..(x0 + tw) * 4];
                let orow = &mut out[y * w + x0..y * w + x0 + tw];
                let keys = &scratch.keys[x0..];
                let att = &scratch.att[x0..x0 + tw];
                let cache = &mut scratch.cache;

                // stage 2: c1 lookups, memo cache first (see
                // `lookup_cache_first`); the memo slot a few pixels ahead
                // is prefetched each step (its address needs only the
                // color bytes; get() falls off the tile end cleanly). The
                // direct[] cell is deliberately not prefetched: it is
                // needed only on a memo miss, and pulling a 1MB table's
                // line per pixel measured slower than the occasional
                // stall.
                for i in 0..tw {
                    if let Some(p) = row.get((i + 8) * 4..(i + 8) * 4 + 4) {
                        self.nearest.prefetch_cache_slot(cache, p[0], p[1], p[2]);
                    }
                    let p = &row[i * 4..i * 4 + 4];
                    scratch.pk1[i] = self
                        .nearest
                        .lookup_cache_first(cache, keys[i], p[0], p[1], p[2]);
                }

                // A fully attenuated tile can't flip any pixel to c2, so
                // stages 3-5 would only reproduce c1 — emit it directly.
                // (Common on busy content, where the gate is doing its job.)
                let tile_live = !gate_on || att.iter().any(|&a| a != 0);

                // stage 3: errors, far-probe colors, and their keys; a
                // tile with all-exact pixels short-circuits the same way
                let tile_live = tile_live
                    && crate::simdops::bn_probe(
                        level,
                        row,
                        &scratch.pk1[..tw],
                        &mut scratch.ors[..tw],
                        &mut scratch.c2c[..tw],
                        &mut scratch.keys2[..tw],
                    );

                if tile_live {
                    // stage 4: c2 lookups (prefetched like stage 2), only
                    // for pixels that are inexact and not fully attenuated:
                    // the threshold pick reduces to c1 for the rest, so
                    // their stale pk2 entry is never consulted
                    for i in 0..tw {
                        if let Some(&c) = scratch.c2c.get(i + 8) {
                            self.nearest.prefetch_cache_slot(
                                cache,
                                (c >> 16) as u8,
                                (c >> 8) as u8,
                                c as u8,
                            );
                        }
                        if (scratch.ors[i] != 0) & (att[i] != 0) {
                            let c = scratch.c2c[i];
                            scratch.pk2[i] = self.nearest.lookup_cache_first(
                                cache,
                                scratch.keys2[i],
                                (c >> 16) as u8,
                                (c >> 8) as u8,
                                c as u8,
                            );
                        }
                    }

                    // stage 5: threshold pick (x0 is a multiple of 64, so
                    // the kernel's local x & 63 mask indexing stays aligned)
                    crate::simdops::bn_threshold(
                        level,
                        row,
                        &scratch.pk1[..tw],
                        &scratch.pk2[..tw],
                        mrow,
                        att,
                        orow,
                    );
                } else {
                    for (o, &p) in orow.iter_mut().zip(&scratch.pk1[..tw]) {
                        *o = p as u8;
                    }
                }

                // stage 6: transparent pixels override whatever was computed
                if has_alpha_row {
                    for (o, px) in orow.iter_mut().zip(row.as_chunks::<4>().0) {
                        if px[3] < 128 {
                            *o = self.trans_idx;
                        }
                    }
                }
                x0 += tw;
            }
            if gate_on {
                // current row becomes the next row's activity reference
                std::mem::swap(&mut scratch.row, &mut scratch.row2);
            }
        }
        has_alpha
    }

    /// `Dither::Auto`: the staged blue-noise pipeline with a per-tile
    /// banding gate (see `BandGate`). Pass A looks up c1 for the whole
    /// frame, storing one index byte per pixel, and scores each tile from
    /// the run structure of the index map; pass B is the ungated pipeline
    /// with the attenuation forced to zero in tiles that do not band, so
    /// their far-candidate lookups are skipped and they emit c1 directly.
    /// The c1 lookups are done once; pass B re-fills rows (a copy or
    /// expand for packed frames) and re-derives the packed colour from the
    /// stored index.
    #[allow(clippy::needless_range_loop)]
    fn quantize_bluenoise_auto(
        &self,
        src: &crate::color::RowSource,
        w: usize,
        h: usize,
        scratch: &mut QuantScratch,
        out: &mut [u8],
        band: &BandGate,
    ) -> bool {
        // 64-px stripes (mask-aligned): dead tiles then skip the probe and
        // threshold stages at tile granularity instead of per 256
        const TILE: usize = 64;
        const T: usize = BandGate::TILE;
        let mask32 = &crate::bluenoise::BLUE_NOISE_64_U32;
        let level = crate::simdops::level();
        let gate_on = self.gate > 0;
        let tx = w.div_ceil(T);
        let ty = h.div_ceil(T);
        if scratch.idx_frame.len() < w * h {
            scratch.idx_frame.resize(w * h, 0);
        }
        scratch.tile_counts.clear();
        scratch.tile_counts.resize(tx * ty, 0);
        scratch.tile_live.clear();
        scratch.tile_live.resize(tx * ty, 0);

        // ---- pass A: c1 per pixel, tile contour counts ----------------
        for y in 0..h {
            let keys_ready = src.fill_row_with_grid_keys(y, &mut scratch.row, &mut scratch.keys);
            if !keys_ready {
                crate::simdops::bn_keys(level, &scratch.row, &mut scratch.keys);
            }
            let cache = &mut scratch.cache;
            let row = &scratch.row[..];
            let (before, rest) = scratch.idx_frame.split_at_mut(y * w);
            let irow = &mut rest[..w];
            for i in 0..w {
                if let Some(p) = row.get((i + 8) * 4..(i + 8) * 4 + 4) {
                    self.nearest.prefetch_cache_slot(cache, p[0], p[1], p[2]);
                }
                let p = &row[i * 4..i * 4 + 4];
                irow[i] = self
                    .nearest
                    .lookup_cache_first(cache, scratch.keys[i], p[0], p[1], p[2])
                    as u8;
            }
            // score: run flags + contour candidates (SIMD), then the sparse
            // colour-pair test on candidates, counted per tile
            const P: usize = crate::simdops::BAND_PAD;
            scratch.idxp[P..P + w].copy_from_slice(irow);
            let prev_row: Option<&[u8]> = if y > 0 {
                Some(&before[(y - 1) * w..])
            } else {
                None
            };
            crate::simdops::band_score_row(
                level,
                &scratch.idxp,
                prev_row,
                &scratch.flat_prev,
                &mut scratch.flat_cur,
                &mut scratch.ltmp,
                &mut scratch.cand,
                w,
            );
            let counts = &mut scratch.tile_counts;
            let trow = (y / T) * tx;
            let pairs = &band.pairs;
            let cand = &scratch.cand;
            // candidates are sparse: skip 8 pixels at a time when none
            let mut i = 0usize;
            while i < w {
                if i + 8 <= w && u64::from_ne_bytes(cand[i..i + 8].try_into().unwrap()) == 0 {
                    i += 8;
                    continue;
                }
                let c = cand[i];
                if c != 0 {
                    let b = irow[i];
                    if c & 1 != 0 && pairs[irow[i - 1] as usize * 256 + b as usize] != 0 {
                        counts[trow + i / T] += 1;
                    }
                    if c & 2 != 0 && pairs[prev_row.unwrap()[i] as usize * 256 + b as usize] != 0 {
                        counts[trow + i / T] += 1;
                    }
                }
                i += 1;
            }
            std::mem::swap(&mut scratch.flat_prev, &mut scratch.flat_cur);
        }
        // tile decision
        let mut live = 0u64;
        for tyy in 0..ty {
            let th = T.min(h - tyy * T);
            for txx in 0..tx {
                let tw = T.min(w - txx * T);
                let area = (tw * th) as u32;
                let l = scratch.tile_counts[tyy * tx + txx] * BandGate::DENSITY > area;
                scratch.tile_live[tyy * tx + txx] = l as u8;
                live += l as u64;
            }
        }
        // a frame that bands almost everywhere is gradient content: dither
        // it whole. Mixed tiles on such frames measured worse than either
        // extreme (LZW loses the pattern's regularity, and seams show).
        if live * BandGate::FULL_DENOM > (tx * ty) as u64 * BandGate::FULL_NUM {
            scratch.tile_live.fill(1);
            live = (tx * ty) as u64;
        }
        band.live_tiles
            .fetch_add(live, std::sync::atomic::Ordering::Relaxed);
        band.total_tiles
            .fetch_add((tx * ty) as u64, std::sync::atomic::Ordering::Relaxed);

        // ---- pass B: staged pipeline, attenuation masked per tile --------
        let mut has_alpha = false;
        for y in 0..h {
            // keys are not needed in pass B (c1 is stored); frames whose
            // rows carry keys for free skip that work
            let keys_ready = src.has_direct_rgb_keys();
            if keys_ready {
                src.fill_row(y, &mut scratch.row);
            } else {
                src.fill_row_with_grid_keys(y, &mut scratch.row, &mut scratch.keys);
            }
            let has_alpha_row = if gate_on && keys_ready {
                let prev: &[u8] = if y == 0 { &scratch.row } else { &scratch.row2 };
                crate::simdops::bn_activity(level, &scratch.row, prev, self.gate, &mut scratch.att);
                false
            } else if gate_on {
                let prev: &[u8] = if y == 0 { &scratch.row } else { &scratch.row2 };
                crate::simdops::bn_keys_att(
                    level,
                    &scratch.row,
                    prev,
                    self.gate,
                    &mut scratch.keys,
                    &mut scratch.att,
                )
            } else {
                scratch.att.fill(256);
                if !keys_ready {
                    crate::simdops::bn_keys(level, &scratch.row, &mut scratch.keys)
                } else {
                    false
                }
            };
            has_alpha |= has_alpha_row;
            let tlive = &scratch.tile_live[(y / T) * tx..(y / T + 1) * tx];
            for (txx, &l) in tlive.iter().enumerate() {
                if l == 0 {
                    let x0 = txx * T;
                    scratch.att[x0..(x0 + T).min(w)].fill(0);
                }
            }
            let mrow: &[u32; 64] = mask32[(y & 63) << 6..((y & 63) << 6) + 64]
                .try_into()
                .unwrap();
            let irow = &scratch.idx_frame[y * w..(y + 1) * w];
            let mut x0 = 0usize;
            while x0 < w {
                let tw = TILE.min(w - x0);
                let row = &scratch.row[x0 * 4..(x0 + tw) * 4];
                let orow = &mut out[y * w + x0..y * w + x0 + tw];
                let att = &scratch.att[x0..x0 + tw];
                let cache = &mut scratch.cache;
                let tile_live = att.iter().any(|&a| a != 0);
                if tile_live {
                    for i in 0..tw {
                        scratch.pk1[i] = self.nearest.packed(irow[x0 + i]);
                    }
                }
                let tile_live = tile_live
                    && crate::simdops::bn_probe(
                        level,
                        row,
                        &scratch.pk1[..tw],
                        &mut scratch.ors[..tw],
                        &mut scratch.c2c[..tw],
                        &mut scratch.keys2[..tw],
                    );
                if tile_live {
                    for i in 0..tw {
                        if let Some(&c) = scratch.c2c.get(i + 8) {
                            self.nearest.prefetch_cache_slot(
                                cache,
                                (c >> 16) as u8,
                                (c >> 8) as u8,
                                c as u8,
                            );
                        }
                        if (scratch.ors[i] != 0) & (att[i] != 0) {
                            let c = scratch.c2c[i];
                            scratch.pk2[i] = self.nearest.lookup_cache_first(
                                cache,
                                scratch.keys2[i],
                                (c >> 16) as u8,
                                (c >> 8) as u8,
                                c as u8,
                            );
                        }
                    }
                    crate::simdops::bn_threshold(
                        level,
                        row,
                        &scratch.pk1[..tw],
                        &scratch.pk2[..tw],
                        mrow,
                        att,
                        orow,
                    );
                } else {
                    orow.copy_from_slice(&irow[x0..x0 + tw]);
                }
                if has_alpha_row {
                    for (o, px) in orow.iter_mut().zip(row.as_chunks::<4>().0) {
                        if px[3] < 128 {
                            *o = self.trans_idx;
                        }
                    }
                }
                x0 += tw;
            }
            if gate_on {
                std::mem::swap(&mut scratch.row, &mut scratch.row2);
            }
        }
        has_alpha
    }

    fn quantize_bluenoise_scalar(
        &self,
        src: &crate::color::RowSource,
        w: usize,
        h: usize,
        scratch: &mut QuantScratch,
        out: &mut [u8],
    ) -> bool {
        // Per-pixel formulation: with an exact palette every pixel takes
        // the zero-error early-out, which no staged pipeline can beat.
        let mask = &crate::bluenoise::BLUE_NOISE_64;
        let mut has_alpha = false;
        let cache = &mut scratch.cache;
        for y in 0..h {
            src.fill_row(y, &mut scratch.row);
            let row = &scratch.row[..];
            let orow = &mut out[y * w..(y + 1) * w];
            let mrow = &mask[(y & 63) << 6..((y & 63) << 6) + 64];
            let pixels = row.as_chunks::<4>().0;
            for (x, (px, o)) in pixels.iter().zip(orow.iter_mut()).enumerate() {
                if let Some(pf) = pixels.get(x + 8) {
                    self.nearest.prefetch(pf[0], pf[1], pf[2]);
                }
                if px[3] < 128 {
                    *o = self.trans_idx;
                    has_alpha = true;
                    continue;
                }
                let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
                let p1 = self.nearest.lookup_packed(cache, px[0], px[1], px[2]);
                let idx1 = p1 as u8;
                let er = r - (p1 >> 24) as i32;
                let eg = g - ((p1 >> 16) & 0xFF) as i32;
                let eb = b - ((p1 >> 8) & 0xFF) as i32;
                if er == 0 && eg == 0 && eb == 0 {
                    *o = idx1;
                    continue;
                }
                // probe across the error direction for the far candidate
                let r2 = (r + 2 * er).clamp(0, 255) as u8;
                let g2 = (g + 2 * eg).clamp(0, 255) as u8;
                let b2 = (b + 2 * eb).clamp(0, 255) as u8;
                let p2 = self.nearest.lookup_packed(cache, r2, g2, b2);
                let idx2 = p2 as u8;
                if idx2 == idx1 {
                    *o = idx1;
                    continue;
                }
                let dr = (p2 >> 24) as i32 - (p1 >> 24) as i32;
                let dg = ((p2 >> 16) & 0xFF) as i32 - ((p1 >> 16) & 0xFF) as i32;
                let db = ((p2 >> 8) & 0xFF) as i32 - ((p1 >> 8) & 0xFF) as i32;
                // fraction of the c1->c2 span covered by p, vs threshold
                let num = (er * dr + eg * dg + eb * db).max(0);
                let den = dr * dr + dg * dg + db * db;
                let m = mrow[x & 63] as i32;
                *o = if m * den < num * 256 { idx2 } else { idx1 };
            }
        }
        has_alpha
    }

    fn quantize_plain(
        &self,
        src: &crate::color::RowSource,
        w: usize,
        h: usize,
        scratch: &mut QuantScratch,
        out: &mut [u8],
    ) -> bool {
        let mut has_alpha = false;
        let cache = &mut scratch.cache;
        for y in 0..h {
            src.fill_row(y, &mut scratch.row);
            let orow = &mut out[y * w..(y + 1) * w];
            let pixels = scratch.row.as_chunks::<4>().0;
            for (x, (px, o)) in pixels.iter().zip(orow.iter_mut()).enumerate() {
                if let Some(pf) = pixels.get(x + 8) {
                    self.nearest.prefetch(pf[0], pf[1], pf[2]);
                }
                if px[3] < 128 {
                    *o = self.trans_idx;
                    has_alpha = true;
                } else {
                    *o = self.nearest.lookup_packed(cache, px[0], px[1], px[2]) as u8;
                }
            }
        }
        has_alpha
    }

    /// 8x8 ordered Bayer: offset each channel by the cell's threshold
    /// before the nearest-color lookup. Pixels the palette already
    /// reproduces exactly are left alone — there is no quantization error
    /// to hide there, so perturbing them would only introduce error and
    /// break up runs the encoder would otherwise keep.
    fn quantize_bayer(
        &self,
        src: &crate::color::RowSource,
        w: usize,
        h: usize,
        scratch: &mut QuantScratch,
        out: &mut [u8],
    ) -> bool {
        // Every source color is in the palette: no pixel has error to
        // dither, so the whole frame takes the plain path.
        if self.exact_palette {
            return self.quantize_plain(src, w, h, scratch, out);
        }
        let mut has_alpha = false;
        let cache = &mut scratch.cache;
        for y in 0..h {
            src.fill_row(y, &mut scratch.row);
            let row = &scratch.row[..];
            let orow = &mut out[y * w..(y + 1) * w];
            let brow = &BAYER8[y & 7];
            let pixels = row.as_chunks::<4>().0;
            for (x, (px, o)) in pixels.iter().zip(orow.iter_mut()).enumerate() {
                if let Some(pf) = pixels.get(x + 8) {
                    self.nearest.prefetch(pf[0], pf[1], pf[2]);
                }
                if px[3] < 128 {
                    *o = self.trans_idx;
                    has_alpha = true;
                    continue;
                }
                // An exact palette match has no error to hide: emit it
                // as-is rather than letting the threshold push it onto a
                // neighboring color (packed color is bits 8..32).
                let c1 = self.nearest.lookup_packed(cache, px[0], px[1], px[2]);
                let rgb = ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | px[2] as u32;
                if (c1 >> 8) == rgb {
                    *o = c1 as u8;
                    continue;
                }
                // Threshold offset in [-8, 8): matches ffmpeg's default
                // bayer_scale=2 ((value >> 2) - 8).
                let t = ((brow[x & 7] as i32) >> 2) - 8;
                let r = (px[0] as i32 + t).clamp(0, 255) as u8;
                let g = (px[1] as i32 + t).clamp(0, 255) as u8;
                let b = (px[2] as i32 + t).clamp(0, 255) as u8;
                *o = self.nearest.lookup_packed(cache, r, g, b) as u8;
            }
        }
        has_alpha
    }

    // the interleaved stripe loop indexes two rows at different offsets;
    // an iterator form would obscure the lag structure
    #[allow(clippy::needless_range_loop)]
    fn quantize_diffuse(
        &self,
        src: &crate::color::RowSource,
        w: usize,
        h: usize,
        scratch: &mut QuantScratch,
        out: &mut [u8],
    ) -> bool {
        let mut has_alpha = false;
        let cache = &mut scratch.cache;
        // Error buffers flowing into rows y, y+1, y+2 (one pad cell each
        // side). Rows are processed in stripes of two, interleaved with the
        // lower row trailing the upper by two columns: pixel (x, y+1) only
        // depends on row-y pixels up to column x+1 (Sierra-2-4A reaches
        // at most one column right on the row below), so the lag keeps the
        // arithmetic — and thus the output — exactly the row-serial result
        // while giving the CPU two independent dependency chains to
        // overlap. Error diffusion's serial chain
        // (clamp -> table lookup -> error -> next pixel) is what bounds
        // this stage, so the second in-flight chain is nearly free.
        let mut err_a: Vec<[i32; 3]> = vec![[0; 3]; w + 2];
        let mut err_b: Vec<[i32; 3]> = vec![[0; 3]; w + 2];
        let mut err_c: Vec<[i32; 3]> = vec![[0; 3]; w + 2];
        let mut row_a = std::mem::take(&mut scratch.row);
        let mut row_b = std::mem::take(&mut scratch.row2);

        macro_rules! step {
            ($x:expr, $pixels:expr, $carry:expr, $cur:ident, $next:ident, $o:expr) => {{
                let x: usize = $x;
                let px = &$pixels[x];
                if let Some(pf) = $pixels.get(x + 8) {
                    self.nearest.prefetch(pf[0], pf[1], pf[2]);
                }
                if px[3] < 128 {
                    $o = self.trans_idx;
                    has_alpha = true;
                    $carry = [0; 3];
                } else {
                    let e = &$cur[x + 1];
                    let r = (px[0] as i32 + $carry[0] + e[0]).clamp(0, 255);
                    let g = (px[1] as i32 + $carry[1] + e[1]).clamp(0, 255);
                    let b = (px[2] as i32 + $carry[2] + e[2]).clamp(0, 255);
                    let p = self.nearest.lookup_packed(cache, r as u8, g as u8, b as u8);
                    $o = p as u8;
                    let er = r - (p >> 24) as i32;
                    let eg = g - ((p >> 16) & 0xFF) as i32;
                    let eb = b - ((p >> 8) & 0xFF) as i32;
                    // Sierra-2-4A: 2/4 right, 1/4 down-left, 1/4 down.
                    // Truncating division (like ffmpeg's
                    // `err*scale/(1<<n)`) — an arithmetic shift would
                    // round negative errors toward -inf and diffuse
                    // more than 100% of the error, which diverges
                    // into noise.
                    $carry = [er / 2, eg / 2, eb / 2];
                    let dl = &mut $next[x];
                    dl[0] += er / 4;
                    dl[1] += eg / 4;
                    dl[2] += eb / 4;
                    let d = &mut $next[x + 1];
                    d[0] += er / 4;
                    d[1] += eg / 4;
                    d[2] += eb / 4;
                }
            }};
        }

        let mut y = 0usize;
        while y < h {
            if y + 1 < h {
                // stripe of two rows, interleaved at a two-column lag
                err_b.iter_mut().for_each(|e| *e = [0; 3]);
                err_c.iter_mut().for_each(|e| *e = [0; 3]);
                src.fill_row(y, &mut row_a);
                src.fill_row(y + 1, &mut row_b);
                let pa = row_a.as_chunks::<4>().0;
                let pb = row_b.as_chunks::<4>().0;
                let (oa, ob) = out[y * w..(y + 2) * w].split_at_mut(w);
                let mut carry_a = [0i32; 3];
                let mut carry_b = [0i32; 3];
                for xa in 0..w + 2 {
                    if xa < w {
                        step!(xa, pa, carry_a, err_a, err_b, oa[xa]);
                    }
                    if xa >= 2 {
                        let xb = xa - 2;
                        step!(xb, pb, carry_b, err_b, err_c, ob[xb]);
                    }
                }
                // errors into row y+2 become the next stripe's incoming set
                std::mem::swap(&mut err_a, &mut err_c);
                y += 2;
            } else {
                // odd final row: plain serial pass
                err_b.iter_mut().for_each(|e| *e = [0; 3]);
                src.fill_row(y, &mut row_a);
                let pa = row_a.as_chunks::<4>().0;
                let orow = &mut out[y * w..(y + 1) * w];
                let mut carry = [0i32; 3];
                for x in 0..w {
                    step!(x, pa, carry, err_a, err_b, orow[x]);
                }
                y += 1;
            }
        }
        scratch.row = row_a;
        scratch.row2 = row_b;
        has_alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::NearestMap;

    #[test]
    fn plain_exact_colors_lossless() {
        let colors = vec![[0u8, 0, 0], [255, 255, 255], [255, 0, 0]];
        let nm = NearestMap::build(&colors);
        let q = Quantizer {
            nearest: &nm,
            trans_idx: 3,
            exact_palette: false,
            gate: 0,
            band: None,
        };
        let rgba = vec![
            255u8, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0,
        ];
        let frame = crate::input::Frame::Rgba(rgba);
        let src = crate::color::RowSource::new(&frame, 4, 1, None);
        let mut out = [9u8; 4];
        let mut scratch = QuantScratch::new(4);
        let has_alpha = q.quantize(&src, 4, 1, Dither::None, &mut scratch, &mut out);
        assert!(has_alpha);
        assert_eq!(out, [2, 0, 1, 3]);
    }

    fn xorshift(x: &mut u32) -> u32 {
        *x ^= *x << 13;
        *x ^= *x >> 17;
        *x ^= *x << 5;
        *x
    }

    fn quantize_rgba(q: &Quantizer, rgba: Vec<u8>, w: usize, h: usize, mode: Dither) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        let frame = crate::input::Frame::Rgba(rgba);
        let src = crate::color::RowSource::new(&frame, w, h, None);
        let mut scratch = QuantScratch::new(w);
        q.quantize(&src, w, h, mode, &mut scratch, &mut out);
        out
    }

    #[test]
    fn gate_preserves_gradients() {
        // A smooth ramp's activity sits below the gate, so gated output
        // must equal ungated output — banding protection is untouched.
        let (w, h) = (64usize, 64usize);
        let mut rgba = Vec::with_capacity(w * h * 4);
        for _y in 0..h {
            for x in 0..w {
                let v = (x * 2) as u8;
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let colors = vec![[0u8, 0, 0], [64, 64, 64], [128, 128, 128]];
        let nm = NearestMap::build(&colors);
        let mk = |gate| Quantizer {
            nearest: &nm,
            trans_idx: 3,
            exact_palette: false,
            gate,
            band: None,
        };
        let ungated = quantize_rgba(&mk(0), rgba.clone(), w, h, Dither::BlueNoise);
        let gated = quantize_rgba(&mk(16), rgba.clone(), w, h, Dither::BlueNoise);
        assert_eq!(ungated, gated);
        // sanity: the ramp actually dithers (both palette neighbors appear
        // in a middle band)
        let band: Vec<u8> = ungated
            .iter()
            .enumerate()
            .filter(|(i, _)| (20..44).contains(&(i % w)))
            .map(|(_, &v)| v)
            .collect();
        assert!(band.contains(&0) && band.contains(&1));
    }

    #[test]
    fn gate_flattens_busy_content() {
        // A 1px checkerboard is maximal activity: every pixel except the
        // top-left corner (which has no left/up neighbor and reads as
        // flat) is fully attenuated and must match the undithered output.
        // Ungated blue-noise, by contrast, flips a substantial share.
        let (w, h) = (64usize, 64usize);
        let mut rgba = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) & 1 == 0 { 64u8 } else { 192 };
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let colors = vec![[0u8, 0, 0], [255, 255, 255]];
        let nm = NearestMap::build(&colors);
        let mk = |gate| Quantizer {
            nearest: &nm,
            trans_idx: 2,
            exact_palette: false,
            gate,
            band: None,
        };
        let plain = quantize_rgba(&mk(0), rgba.clone(), w, h, Dither::None);
        let gated = quantize_rgba(&mk(16), rgba.clone(), w, h, Dither::BlueNoise);
        let ungated = quantize_rgba(&mk(0), rgba.clone(), w, h, Dither::BlueNoise);
        assert_eq!(gated[1..], plain[1..]);
        let flips = ungated.iter().zip(&plain).filter(|(a, b)| a != b).count();
        assert!(flips > w * h / 10, "expected heavy dither, got {flips}");
    }

    #[test]
    fn staged_matches_scalar_reference() {
        // The staged SIMD pipeline (odd width -> tail lanes; gate on ->
        // attenuation, row ping-pong, tile fast paths) must reproduce the
        // per-pixel formulation exactly.
        let (w, h) = (77usize, 40usize);
        let gate = 16u32;
        let mut seed = 0xDEADBEEFu32;
        let mut rgba = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let r = xorshift(&mut seed);
                if y >= 30 && r.is_multiple_of(23) {
                    rgba.extend_from_slice(&[0, 0, 0, 0]); // transparent
                    continue;
                }
                let px = if y < 10 {
                    // smooth ramp: full dither
                    let v = (x * 3) as u8;
                    [v, v.wrapping_add(20), v, 255]
                } else if y < 20 {
                    // moderate texture: partial attenuation
                    let b = (x * 2) as u8;
                    [
                        b.wrapping_add((r & 7) as u8),
                        b.wrapping_add(((r >> 3) & 7) as u8),
                        b,
                        255,
                    ]
                } else {
                    // heavy noise: fully attenuated
                    [
                        (r & 0xFF) as u8,
                        ((r >> 8) & 0xFF) as u8,
                        ((r >> 16) & 0xFF) as u8,
                        255,
                    ]
                };
                rgba.extend_from_slice(&px);
            }
        }
        let mut colors = Vec::new();
        for i in 0..13u32 {
            let c = xorshift(&mut seed);
            let _ = i;
            colors.push([
                (c & 0xFF) as u8,
                ((c >> 8) & 0xFF) as u8,
                ((c >> 16) & 0xFF) as u8,
            ]);
        }
        let nm = NearestMap::build(&colors);
        let trans_idx = colors.len() as u8;
        let q = Quantizer {
            nearest: &nm,
            trans_idx,
            exact_palette: false,
            gate,
            band: None,
        };
        let got = quantize_rgba(&q, rgba.clone(), w, h, Dither::BlueNoise);

        // scalar reference
        let mask = &crate::bluenoise::BLUE_NOISE_64;
        let mut cache = crate::palette::IdxCache::default();
        let mut want = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 4;
                let px = &rgba[o..o + 4];
                if px[3] < 128 {
                    want[y * w + x] = trans_idx;
                    continue;
                }
                let cur = &rgba[y * w * 4..(y + 1) * w * 4];
                let prev = &rgba[y.saturating_sub(1) * w * 4..][..w * 4];
                let att = crate::simdops::att_scalar(cur, prev, x, gate as i32 + 64) as i32;
                let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
                let p1 = nm.lookup_packed(&mut cache, px[0], px[1], px[2]);
                let idx1 = p1 as u8;
                let er = r - (p1 >> 24) as i32;
                let eg = g - ((p1 >> 16) & 0xFF) as i32;
                let eb = b - ((p1 >> 8) & 0xFF) as i32;
                if er == 0 && eg == 0 && eb == 0 {
                    want[y * w + x] = idx1;
                    continue;
                }
                let r2 = (r + 2 * er).clamp(0, 255) as u8;
                let g2 = (g + 2 * eg).clamp(0, 255) as u8;
                let b2 = (b + 2 * eb).clamp(0, 255) as u8;
                let p2 = nm.lookup_packed(&mut cache, r2, g2, b2);
                let dr = (p2 >> 24) as i32 - (p1 >> 24) as i32;
                let dg = ((p2 >> 16) & 0xFF) as i32 - ((p1 >> 16) & 0xFF) as i32;
                let db = ((p2 >> 8) & 0xFF) as i32 - ((p1 >> 8) & 0xFF) as i32;
                let num = (er * dr + eg * dg + eb * db).max(0);
                let den = dr * dr + dg * dg + db * db;
                let m = mask[((y & 63) << 6) + (x & 63)] as i32;
                want[y * w + x] = if m * den < num * att { p2 as u8 } else { idx1 };
            }
        }
        assert_eq!(got, want);
    }

    #[test]
    fn diffuse_no_error_when_exact() {
        // With an exact palette, dithering must be a no-op.
        let colors = vec![[10u8, 20, 30], [200, 100, 50]];
        let nm = NearestMap::build(&colors);
        let q = Quantizer {
            nearest: &nm,
            trans_idx: 2,
            exact_palette: false,
            gate: 0,
            band: None,
        };
        let mut rgba = Vec::new();
        for i in 0..64 {
            let c = &colors[i % 2];
            rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
        let rgba = rgba;
        let mut out = vec![0u8; 64];
        let frame = crate::input::Frame::Rgba(rgba);
        let src = crate::color::RowSource::new(&frame, 8, 8, None);
        let mut scratch = QuantScratch::new(8);
        q.quantize(&src, 8, 8, Dither::Sierra2_4a, &mut scratch, &mut out);
        for (i, &o) in out.iter().enumerate() {
            assert_eq!(o as usize, i % 2);
        }
    }

    /// Bayer must leave pixels the palette reproduces exactly alone:
    /// there is no quantization error to hide, so offsetting them before
    /// the lookup only moves them onto a neighboring color (and breaks up
    /// runs the encoder would otherwise keep).
    #[test]
    fn bayer_keeps_exact_matches() {
        // 8 grays, 32 apart: an off-palette midpoint sits between two.
        let colors: Vec<[u8; 3]> = (0..8u8).map(|i| [i * 32, i * 32, i * 32]).collect();
        let nm = NearestMap::build(&colors);
        let (w, h) = (64usize, 8usize);
        // even columns are exactly a palette color, odd columns sit
        // halfway between two of them
        let mut rgba = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let step = ((x / 2 + y) % 8) as u8;
                let v = if x % 2 == 0 {
                    step * 32
                } else {
                    step.min(6) * 32 + 16
                };
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let q = Quantizer {
            nearest: &nm,
            trans_idx: 8,
            exact_palette: false,
            gate: 0,
            band: None,
        };
        let out = quantize_rgba(&q, rgba.clone(), w, h, Dither::Bayer);
        for y in 0..h {
            for x in (0..w).step_by(2) {
                let want = ((x / 2 + y) % 8) as u8;
                assert_eq!(
                    out[y * w + x],
                    want,
                    "exact pixel at ({x}, {y}) was dithered off its own color"
                );
            }
        }
        // ...while the off-palette columns still dither, so the exact
        // check has not simply switched dithering off: the same frame
        // differs from the undithered result.
        let plain = quantize_rgba(&q, rgba, w, h, Dither::None);
        assert_ne!(out, plain, "bayer stopped dithering off-palette pixels");

        // and a uniform color halfway between two palette entries lands
        // on both of them, per mask cell
        let v = 112u8; // between palette entries 3 (96) and 4 (128)
        let flat: Vec<u8> = std::iter::repeat_n([v, v, v, 255], 64).flatten().collect();
        let mut got = quantize_rgba(&q, flat, 8, 8, Dither::Bayer);
        got.sort_unstable();
        got.dedup();
        assert_eq!(got, [3, 4]);
    }

    /// The whole-frame shortcut: when every source color is in the
    /// palette, bayer is the plain nearest-color path.
    #[test]
    fn bayer_exact_palette_is_lossless() {
        // grays 4 apart: closer together than the mask's +/-8 offset, so
        // an offset lookup would land on a neighboring entry (this is the
        // shape of a soft gradient that fits in the palette exactly)
        let colors: Vec<[u8; 3]> = (0..64u8).map(|i| [i * 4, i * 4, i * 4]).collect();
        let nm = NearestMap::build(&colors);
        let (w, h) = (16usize, 16usize);
        let mut rgba = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let c = colors[(x + y) % 64];
                rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        let q = Quantizer {
            nearest: &nm,
            trans_idx: 64,
            exact_palette: true,
            gate: 0,
            band: None,
        };
        let out = quantize_rgba(&q, rgba.clone(), w, h, Dither::Bayer);
        let plain = quantize_rgba(&q, rgba, w, h, Dither::None);
        assert_eq!(out, plain);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(out[y * w + x] as usize, (x + y) % 64);
            }
        }
    }
}
