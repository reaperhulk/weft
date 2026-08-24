//! Per-frame quantization to palette indices, with optional error-diffusion
//! or ordered dithering. Frames are independent, so this stage parallelizes
//! across frames; error diffusion stays serial only *within* a frame.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dither {
    /// Sierra-2-4A ("filter lite"): ffmpeg paletteuse's default.
    Sierra2_4a,
    /// Floyd–Steinberg.
    FloydSteinberg,
    /// 8x8 ordered Bayer: fastest dithered mode, fully branch-predictable.
    Bayer,
    /// 64x64 void-and-cluster blue-noise two-candidate ordered dither: no
    /// serial error-diffusion chain, temporally stable, far less visible
    /// structure than Bayer.
    BlueNoise,
    None,
}

/// Minimum percentage of unchanged pixels (sampled) for the reuse path to
/// pay for its change masks.
pub const REUSE_MIN_PCT: usize = 50;

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

/// True when every one of the `n` bits from `start` is set. Tiles start
/// on a multiple of 64 pixels, so this is a whole number of words except
/// in the row's last, short tile.
#[inline(always)]
fn all_set(bits: &[u64], start: usize, n: usize) -> bool {
    debug_assert_eq!(start % 64, 0);
    let w0 = start >> 6;
    let full = n >> 6;
    if !bits[w0..w0 + full].iter().all(|&v| v == !0) {
        return false;
    }
    let rest = n & 63;
    rest == 0 || bits[w0 + full] & ((1u64 << rest) - 1) == (1u64 << rest) - 1
}

/// The 24-bit colour of an RGBA pixel, in the `r<<16 | g<<8 | b` packing
/// the nearest-map's memo cache and probe colours use.
#[inline(always)]
fn rgb24(p: &[u8]) -> u32 {
    ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32
}

/// The previous frame's source and quantized indices, for reusing the
/// indices of pixels whose quantization cannot have changed.
#[derive(Clone, Copy)]
pub struct Reuse<'a> {
    pub prev_src: &'a crate::input::Frame,
    pub prev_idx: &'a [u8],
}

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
    wl: Vec<u32>,
    /// Per-pixel changed-vs-previous-frame flags for this row and the one above.
    /// One bit per pixel: does it differ from the previous frame? For
    /// this row and the one above.
    chg: Vec<u64>,
    chg_prev: Vec<u64>,
    /// One bit per pixel: can its index be reused verbatim?
    sf: Vec<u64>,
    wl2: Vec<u32>,
    att: Vec<u32>,
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
            wl: vec![0; w],
            chg: vec![!0u64; w.div_ceil(64)],
            chg_prev: vec![!0u64; w.div_ceil(64)],
            sf: vec![0u64; w.div_ceil(64)],
            wl2: vec![0; w],
            // 256 = no attenuation: with the gate off this is never
            // rewritten and the threshold pick reduces to the ungated one
            att: vec![256; w],
        }
    }
}

impl<'a> Quantizer<'a> {
    /// Quantize a frame (accessed row-by-row via `src`, so YUV conversion
    /// stays fused and cache-resident) into palette indices. Returns true
    /// if any pixel was alpha-transparent.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn quantize(
        &self,
        src: &crate::color::RowSource,
        w: usize,
        h: usize,
        mode: Dither,
        scratch: &mut QuantScratch,
        out: &mut [u8],
        reuse: Option<Reuse>,
    ) -> bool {
        self.quantize_band(src, w, h, 0..h, mode, scratch, out, reuse)
    }

    /// Whether `quantize_band` can be given a partial row range for this
    /// mode. Error diffusion carries state along the whole frame, and the
    /// exact-palette and undithered modes have no row structure to split.
    pub fn bandable(&self, mode: Dither) -> bool {
        mode == Dither::BlueNoise && !self.exact_palette
    }

    /// `quantize`, restricted to rows `rows` (the full frame unless
    /// `bandable`); `out` covers just those rows. Bands of one frame are independent — the blue-noise mode's
    /// only cross-row input is the row above, which a band re-derives for
    /// its first row — so this is the unit the quantize passes
    /// parallelize over: whole frames leave workers idle at each wave
    /// boundary (measured 65-92% efficiency), bands do not.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_band(
        &self,
        src: &crate::color::RowSource,
        w: usize,
        h: usize,
        rows: std::ops::Range<usize>,
        mode: Dither,
        scratch: &mut QuantScratch,
        out: &mut [u8],
        reuse: Option<Reuse>,
    ) -> bool {
        match mode {
            Dither::None => self.quantize_plain(src, w, h, scratch, out),
            Dither::Bayer => self.quantize_bayer(src, w, h, scratch, out),
            Dither::BlueNoise => self.quantize_bluenoise(src, w, h, rows, scratch, out, reuse),
            Dither::Sierra2_4a => self.quantize_diffuse(src, w, h, scratch, out, false),
            Dither::FloydSteinberg => self.quantize_diffuse(src, w, h, scratch, out, true),
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
    /// SIMD key computation, a branchless gather of the packed fast-path
    /// entries, cache/resolve only for a compacted worklist of misses,
    /// SIMD probe-color math, the same gather/resolve pair for the far
    /// candidates (skipping exact and fully attenuated pixels), and a
    /// SIMD threshold pick. Results are identical to the per-pixel
    /// formulation — the lookups and integer math are the same, just
    /// reordered. Tiles that can't dither at all — every pixel exact, or
    /// every pixel fully attenuated — skip the far-candidate stages and
    /// emit c1 directly.
    // the gather loops index several parallel stage arrays plus a
    // compacting worklist cursor; an iterator form would obscure that
    #[allow(clippy::needless_range_loop)]
    #[allow(clippy::too_many_arguments)]
    fn quantize_bluenoise(
        &self,
        src: &crate::color::RowSource,
        w: usize,
        h: usize,
        rows: std::ops::Range<usize>,
        scratch: &mut QuantScratch,
        out: &mut [u8],
        reuse: Option<Reuse>,
    ) -> bool {
        if self.exact_palette {
            return self.quantize_bluenoise_scalar(src, w, h, scratch, out);
        }
        let y_first = rows.start;
        // Tile width: multiple of 64 so the blue-noise mask stays aligned
        // to tile starts, small enough that a tile's stage arrays (~28
        // bytes per pixel) stay L1-resident between passes.
        const TILE: usize = 256;
        // `rgb << 8 | idx` per palette index, for turning a reused index
        // straight back into the pair the threshold pick expects.
        let pal = self.nearest.packed_palette();
        let mask32 = &crate::bluenoise::BLUE_NOISE_64_U32;
        let level = crate::simdops::level();
        let gate_on = self.gate > 0;
        // Sample every 16th row before committing to the reuse path: on
        // content where most pixels change every frame, building the
        // change masks costs more than the lookups it saves. The sample
        // is 1/16th of that cost and is a deterministic function of the
        // two frames, so it cannot make the output depend on scheduling.
        let reuse = reuse.filter(|r| {
            crate::color::unchanged_pct(r.prev_src, src.frame(), w, h, src.chroma())
                >= REUSE_MIN_PCT
        });
        let mut has_alpha = false;
        if reuse.is_none() {
            // no predecessor: nothing is reusable, and the flags persist
            // across frames in the shared scratch
            scratch.sf.fill(0);
        }
        // A band picks up where the row above it left off: the activity
        // gate reads the previous row's pixels, and the reuse test reads
        // its change mask, so both are re-derived once at the band's top
        // edge (row 0 gates against itself and needs neither).
        if y_first > 0 {
            src.fill_row(y_first - 1, &mut scratch.row2);
            if let Some(r) = reuse {
                crate::color::changed_row(
                    r.prev_src,
                    src.frame(),
                    w,
                    h,
                    src.chroma(),
                    y_first - 1,
                    &mut scratch.chg,
                );
            }
        }
        for y in rows {
            // Which pixels of this row differ from the previous frame's.
            // A pixel whose own source colour, its left neighbour and the
            // pixel above it are all unchanged quantizes to exactly the
            // index it did last frame: the two-candidate pick reads only
            // the colour, the blue-noise mask (fixed per position) and
            // the activity gate (which reads those two neighbours). Such
            // pixels skip both random lookups entirely.
            if let Some(r) = reuse {
                std::mem::swap(&mut scratch.chg, &mut scratch.chg_prev);
                crate::color::changed_row(
                    r.prev_src,
                    src.frame(),
                    w,
                    h,
                    src.chroma(),
                    y,
                    &mut scratch.chg,
                );
                if y == 0 {
                    // row 0 gates against itself, so it has no upper
                    // neighbour to invalidate it
                    scratch.chg_prev.copy_from_slice(&scratch.chg);
                }
            }
            src.fill_row(y, &mut scratch.row);
            // stage 1 runs row-wide: grid keys + alpha presence, with the
            // activity attenuation fused into the same pass over the
            // pixels when the gate is on (row 0 has no upper neighbor, so
            // it gates on horizontal activity alone)
            let has_alpha_row = if gate_on {
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
                crate::simdops::bn_keys(level, &scratch.row, &mut scratch.keys)
            };
            // A pixel is reusable when its own source colour is unchanged
            // and so are the two neighbours the activity gate reads. With
            // the gate off only its own colour matters.
            if reuse.is_some() {
                let (chg, sf) = (&scratch.chg, &mut scratch.sf);
                if gate_on {
                    // shifting the row mask left by one pixel folds in the
                    // left neighbour, carrying across word boundaries
                    let up = &scratch.chg_prev;
                    let mut carry = 0u64;
                    for i in 0..sf.len() {
                        let c = chg[i];
                        sf[i] = !(c | (c << 1) | carry | up[i]);
                        carry = c >> 63;
                    }
                } else {
                    for (o, &c) in sf.iter_mut().zip(chg.iter()) {
                        *o = !c;
                    }
                }
            }
            has_alpha |= has_alpha_row;
            let mrow: &[u32; 64] = mask32[(y & 63) << 6..((y & 63) << 6) + 64]
                .try_into()
                .unwrap();
            let mut x0 = 0usize;
            while x0 < w {
                let tw = TILE.min(w - x0);
                let row = &scratch.row[x0 * 4..(x0 + tw) * 4];
                let orow = &mut out[(y - y_first) * w + x0..(y - y_first) * w + x0 + tw];
                let keys = &scratch.keys[x0..];
                let att = &scratch.att[x0..x0 + tw];
                let sf = &scratch.sf[..];
                let prev_row = reuse.map(|r| &r.prev_idx[y * w + x0..y * w + x0 + tw]);
                let cache = &mut scratch.cache;

                // stage 2: memo-cache probe for every pixel, misses
                // compacted into the worklist. The cache is keyed by the
                // exact colour and hits on 75-95% of pixels, so the 1MB
                // grid table is only touched for the rest — the old order
                // (grid first, cache only for multi-candidate cells) paid
                // two random loads for the common pixel, since 80-95% of
                // pixels land on a multi-candidate cell.
                // A tile every one of whose pixels is reusable is just a
                // copy of last frame's indices: none of the staged passes
                // would compute anything else.
                if let Some(prev_row) = prev_row {
                    if all_set(sf, x0, tw) {
                        orow.copy_from_slice(prev_row);
                        x0 += tw;
                        continue;
                    }
                }
                let mut m1 = 0usize;
                match prev_row {
                    // Reusable pixels — index unchanged from last frame —
                    // skip both lookups and hand the threshold pick two
                    // identical candidates. The two loops are written out
                    // rather than sharing a per-pixel test, which cost a
                    // few percent on the frames that have no predecessor.
                    Some(prev_row) => {
                        for i in 0..tw {
                            if i + 16 < tw {
                                self.nearest
                                    .prefetch_color_slot(cache, rgb24(&row[(i + 16) * 4..]));
                            }
                            if (sf[(x0 + i) >> 6] >> ((x0 + i) & 63)) & 1 != 0 {
                                scratch.pk1[i] = pal[prev_row[i] as usize];
                                continue;
                            }
                            let hit = self.nearest.cache_probe(cache, rgb24(&row[i * 4..]));
                            scratch.pk1[i] = hit.unwrap_or(0);
                            scratch.wl[m1] = i as u32;
                            m1 += hit.is_none() as usize;
                        }
                    }
                    None => {
                        for i in 0..tw {
                            if i + 16 < tw {
                                self.nearest
                                    .prefetch_color_slot(cache, rgb24(&row[(i + 16) * 4..]));
                            }
                            let hit = self.nearest.cache_probe(cache, rgb24(&row[i * 4..]));
                            scratch.pk1[i] = hit.unwrap_or(0);
                            scratch.wl[m1] = i as u32;
                            m1 += hit.is_none() as usize;
                        }
                    }
                }
                // stage 3: resolve the misses through the grid table
                // (prefetching the cell a few misses ahead hides the load)
                for j in 0..m1 {
                    if let Some(&fu) = scratch.wl[..m1].get(j + 8) {
                        self.nearest.prefetch_key(keys[fu as usize]);
                    }
                    let i = scratch.wl[j] as usize;
                    scratch.pk1[i] =
                        self.nearest
                            .resolve_keyed(cache, keys[i], rgb24(&row[i * 4..]));
                }

                // A fully attenuated tile can't flip any pixel to c2, so
                // stages 4-7 would only reproduce c1 — emit it directly.
                // (Common on busy content, where the gate is doing its job.)
                let tile_live = !gate_on || att.iter().any(|&a| a != 0);

                // stage 4: errors, far-probe colors, and their keys; a
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
                    // stage 5a: compact the pixels that can actually
                    // flip to c2 — an exact match has no error to dither
                    // and a fully gated pixel keeps c1 whatever c2 is.
                    // On busy content the gate rules out a third to a
                    // half of the tile, and every one of those was
                    // costing a random memo-cache probe. Pixels not on
                    // the list keep c1 as their c2, which makes the
                    // threshold pick keep c1 (identical candidates).
                    scratch.pk2[..tw].copy_from_slice(&scratch.pk1[..tw]);
                    let mut nlive = 0usize;
                    if prev_row.is_some() {
                        // reused pixels keep c1 as c2, so they never enter
                        for i in 0..tw {
                            scratch.wl2[nlive] = i as u32;
                            let reusable = (sf[(x0 + i) >> 6] >> ((x0 + i) & 63)) & 1;
                            nlive +=
                                ((scratch.ors[i] != 0) & (att[i] != 0) & (reusable == 0)) as usize;
                        }
                    } else {
                        for i in 0..tw {
                            scratch.wl2[nlive] = i as u32;
                            nlive += ((scratch.ors[i] != 0) & (att[i] != 0)) as usize;
                        }
                    }
                    // stage 5b: c2 cache probe over the live list
                    let mut m2 = 0usize;
                    for j in 0..nlive {
                        if let Some(&fu) = scratch.wl2[..nlive].get(j + 16) {
                            self.nearest
                                .prefetch_color_slot(cache, scratch.c2c[fu as usize]);
                        }
                        let i = scratch.wl2[j] as usize;
                        let hit = self.nearest.cache_probe(cache, scratch.c2c[i]);
                        if let Some(p) = hit {
                            scratch.pk2[i] = p;
                        }
                        scratch.wl[m2] = i as u32;
                        m2 += hit.is_none() as usize;
                    }
                    // stage 6: resolve c2 misses (same lookahead as stage 3)
                    for j in 0..m2 {
                        if let Some(&fu) = scratch.wl[..m2].get(j + 8) {
                            self.nearest.prefetch_key(scratch.keys2[fu as usize]);
                        }
                        let i = scratch.wl[j] as usize;
                        scratch.pk2[i] =
                            self.nearest
                                .resolve_keyed(cache, scratch.keys2[i], scratch.c2c[i]);
                    }

                    // stage 7: threshold pick (x0 is a multiple of 64, so
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

                // stage 8: transparent pixels override whatever was computed
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

    fn quantize_bayer(
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
        fs: bool,
    ) -> bool {
        let mut has_alpha = false;
        let cache = &mut scratch.cache;
        // Error buffers flowing into rows y, y+1, y+2 (one pad cell each
        // side). Rows are processed in stripes of two, interleaved with the
        // lower row trailing the upper by two columns: pixel (x, y+1) only
        // depends on row-y pixels up to column x+1 (both Sierra-2-4A and
        // Floyd-Steinberg reach at most one column right on the row below),
        // so the lag keeps the arithmetic — and thus the output — exactly
        // the row-serial result while giving the CPU two independent
        // dependency chains to overlap. Error diffusion's serial chain
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
                    if fs {
                        // Floyd–Steinberg: 7/16 right, 3/16 down-left,
                        // 5/16 down, 1/16 down-right
                        $carry = [er * 7 / 16, eg * 7 / 16, eb * 7 / 16];
                        let dl = &mut $next[x];
                        dl[0] += er * 3 / 16;
                        dl[1] += eg * 3 / 16;
                        dl[2] += eb * 3 / 16;
                        let d = &mut $next[x + 1];
                        d[0] += er * 5 / 16;
                        d[1] += eg * 5 / 16;
                        d[2] += eb * 5 / 16;
                        let dr = &mut $next[x + 2];
                        dr[0] += er / 16;
                        dr[1] += eg / 16;
                        dr[2] += eb / 16;
                    } else {
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
        };
        let rgba = vec![
            255u8, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0,
        ];
        let frame = crate::input::Frame::Rgba(rgba);
        let src = crate::color::RowSource::new(&frame, 4, 1, None);
        let mut out = [9u8; 4];
        let mut scratch = QuantScratch::new(4);
        let has_alpha = q.quantize(&src, 4, 1, Dither::None, &mut scratch, &mut out, None);
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
        q.quantize(&src, w, h, mode, &mut scratch, &mut out, None);
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
        q.quantize(&src, 8, 8, Dither::Sierra2_4a, &mut scratch, &mut out, None);
        for (i, &o) in out.iter().enumerate() {
            assert_eq!(o as usize, i % 2);
        }
    }
}
