//! GIF-flavor LZW encoder, tuned for throughput: open-addressed hash table
//! packed into 32-bit entries small enough to stay L1-resident, 64-bit bit
//! accumulator, sub-block chunking done in a single trailing pass.

const MAX_CODE: u32 = 4096;
/// GIF caps codes at 12 bits; the dictionary stops growing there.
const MAX_WIDTH: u32 = 12;
const TABLE_BITS: u32 = 13;
const TABLE_SIZE: usize = 1 << TABLE_BITS;

// Deferred-clear policy (ported from gifsicle): when the dictionary fills,
// keep using it as long as it compresses well, and emit a clear only when
// the average match length (EWMA of run lengths) degrades — either in
// absolute terms or relative to how much image remains.
const RUN_EWMA_SHIFT: u32 = 4;
const RUN_EWMA_SCALE: u32 = 19;
const RUN_INV_THRESH: u32 = (1 << RUN_EWMA_SCALE) / 3000;

#[inline(always)]
fn update_run_ewma(ewma: &mut u32, run: u32) {
    let r = (run << RUN_EWMA_SCALE) + (1 << (RUN_EWMA_SHIFT - 1));
    if r < *ewma {
        *ewma -= (*ewma - r) >> RUN_EWMA_SHIFT;
    } else {
        *ewma += (r - *ewma) >> RUN_EWMA_SHIFT;
    }
}

#[inline(always)]
fn should_clear(run_ewma: u32, pixels_left: usize, min_code_bits: u32) -> bool {
    run_ewma < ((36 << RUN_EWMA_SCALE) / min_code_bits)
        || pixels_left > (u32::MAX / RUN_INV_THRESH) as usize
        || (run_ewma as u64) < pixels_left as u64 * RUN_INV_THRESH as u64
}

/// Support table for lossy encoding, a port of gifsicle's `--lossy`
/// algorithm: at each match, a DFS over the dictionary trie finds the
/// longest continuation whose per-pixel color error stays within
/// `max_diff` (= level * 10, RGB squared units, gifsicle's scale). Each
/// substitution's signed error feeds forward into the next pixel's
/// comparison at 3/4 decay, so consecutive substitutions cancel out and
/// average color is preserved along runs.
pub struct LossyMap {
    starts: Vec<u32>, // len = 257
    /// Substitution candidates, `static_sq_distance << 8 | index`, sorted
    /// ascending. Carrying the distance lets the DFS stop scanning a list
    /// once the remaining entries are provably over the error cap (see
    /// `cand_limit`) instead of computing a dithered diff for each.
    cands: Vec<u32>,
    /// Per-symbol 256-bit membership mask of that symbol's candidate set
    /// (4 x u64 per symbol, same indices as `cands`). ANDing it with a trie
    /// node's `child_bits` answers "does this node have any child the DFS
    /// would consider substituting?" in four instructions, instead of one
    /// bit test per candidate. Measured on the cghmc corpus, 94% of the
    /// old scan's iterations were bit tests that failed.
    cand_mask: Vec<u64>,
    colors: Vec<[i32; 3]>,
    trans_idx: u8,
    max_diff: u32,
}

/// Bounds candidate fan-out per trie node during the DFS.
const LOSSY_MAX_CANDS: usize = 16;

impl LossyMap {
    /// `level` matches gifsicle's `--lossy N` scale (per-pixel error cap
    /// of N*10 in summed squared RGB units).
    pub fn build(colors: &[[u8; 3]], trans_idx: u8, level: u32) -> Self {
        let max_diff = level * 10;
        // Substitution candidates are gathered within twice the cap radius:
        // the dither feedback can shift the effective comparison point, so
        // colors outside the static cap may still pass at DFS time.
        let radius2 = (max_diff * 4) as i32;
        let ci: Vec<[i32; 3]> = colors
            .iter()
            .map(|c| [c[0] as i32, c[1] as i32, c[2] as i32])
            .collect();
        let mut starts = Vec::with_capacity(257);
        let mut cands = Vec::new();
        let mut cand_mask = vec![0u64; 256 * 4];
        for i in 0..256usize {
            starts.push(cands.len() as u32);
            if i >= ci.len() {
                continue; // transparent slot and padding: no candidates
            }
            let mut list: Vec<(u8, i32)> = ci
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(j, c)| {
                    let d = (ci[i][0] - c[0]).pow(2)
                        + (ci[i][1] - c[1]).pow(2)
                        + (ci[i][2] - c[2]).pow(2);
                    (j as u8, d)
                })
                .filter(|&(_, d)| d <= radius2)
                .collect();
            list.sort_by_key(|&(_, d)| d);
            list.truncate(LOSSY_MAX_CANDS);
            // Packed so the sort order above is the order of the packed
            // values too: `d` dominates, and ties keep index order, which
            // is what the stable sort by `d` produced.
            for &(j, _) in &list {
                cand_mask[i * 4 + (j >> 6) as usize] |= 1u64 << (j & 63);
            }
            cands.extend(list.iter().map(|&(j, d)| ((d as u32) << 8) | j as u32));
        }
        starts.push(cands.len() as u32);
        LossyMap {
            starts,
            cands,
            cand_mask,
            colors: ci,
            trans_idx,
            max_diff,
        }
    }

    /// The symbol's candidate-set membership mask (see `cand_mask`).
    #[inline(always)]
    fn cand_mask(&self, symbol: u8) -> &[u64] {
        &self.cand_mask[symbol as usize * 4..symbol as usize * 4 + 4]
    }

    #[inline(always)]
    fn candidates(&self, symbol: u8) -> &[u32] {
        let s = self.starts[symbol as usize] as usize;
        let e = self.starts[symbol as usize + 1] as usize;
        &self.cands[s..e]
    }

    /// Static squared distance past which no candidate can come in under
    /// `cap`, so a distance-sorted list can stop there.
    ///
    /// With `u` the colour difference and `v` the carried dither, both
    /// views `diff` takes are at least `(|u| - |v|)^2` once `|u| >= |v|`,
    /// so a candidate passes only if `|u| <= |v| + sqrt(cap)`. Testing
    /// `|u|^2 <= 2(|v|^2 + cap)` is weaker than that bound — it never cuts
    /// a candidate the exact test would keep — and needs no square roots.
    #[inline(always)]
    fn cand_limit(dither: &[i32; 3], cap: u32) -> u32 {
        let vn2 = (dither[0] * dither[0] + dither[1] * dither[1] + dither[2] * dither[2]) as u32;
        2 * (vn2 + cap)
    }

    /// gifsicle color_diff: squared error between wanted and written,
    /// taking the smaller of the fully-dithered and half-dithered views
    /// (dithering is opportunistic, not required).
    #[inline(always)]
    fn diff(&self, want: u8, got: u8, dither: &[i32; 3]) -> u32 {
        let a = &self.colors[want as usize];
        let b = &self.colors[got as usize];
        // i32 is enough: the carried dither is a decaying series bounded by
        // 4*255, so each term stays under 1275^2 and the sum under 4.9e6.
        let mut dith = 0i32;
        let mut undith = 0i32;
        for c in 0..3 {
            let base = a[c] - b[c];
            let f = base + dither[c];
            let h = base + dither[c] / 2;
            dith += f * f;
            undith += h * h;
        }
        dith.min(undith) as u32
    }

    /// gifsicle diffused_difference: error carried into the next pixel.
    #[inline(always)]
    fn next_dither(&self, want: u8, got: u8, dither: &[i32; 3]) -> [i32; 3] {
        if want == self.trans_idx || got == self.trans_idx {
            return [0; 3];
        }
        let a = &self.colors[want as usize];
        let b = &self.colors[got as usize];
        [
            a[0] - b[0] + dither[0] * 3 / 4,
            a[1] - b[1] + dither[1] * 3 / 4,
            a[2] - b[2] + dither[2] * 3 / 4,
        ]
    }
}

/// The parts of a lossy DFS that never change during one search. Passing
/// them as one reference instead of six arguments keeps the recursion's
/// stack frame small: the search recurses once per matched pixel, so what
/// each level spills is paid on every visited node.
struct DfsCtx<'a> {
    gen: u16,
    data: &'a [u8],
    map: &'a LossyMap,
    scale: Option<&'a [u8]>,
    visits: u32,
    best: LossyBest,
}

struct LossyBest {
    code: u32,
    end: usize,
    diff: u64,
}

/// Reusable encoder state so per-frame allocations amortize away when a
/// thread encodes many frames.
pub struct LzwEncoder {
    // entry: [key:20 | code:12] packed into u32 (key is 12-bit prefix code
    // + 8-bit appended byte). Codes handed out start at eoi+1 >= 3, so a
    // zero code field marks an empty slot and a clear is one 32KB memset —
    // the whole table stays L1-resident, where the previous generation-
    // stamped u64 table (256KB) bounced through L2 on every probe.
    table: Vec<u32>,
    gen: u16,
    scratch: Vec<u8>,
    // Lossy-only: per-prefix-code bitmap of which symbols continue it in
    // the dictionary (4 x u64 per code), generation-stamped per code so a
    // dictionary clear costs nothing. Most trie nodes have no or one
    // child, and most substitution candidates aren't children, so the DFS
    // tests a bit instead of paying a hash probe (and a diff) per
    // candidate.
    child_gen: Vec<u16>,
    child_bits: Vec<u64>,
}

impl Default for LzwEncoder {
    fn default() -> Self {
        Self {
            table: vec![0; TABLE_SIZE],
            gen: 0,
            scratch: Vec::new(),
            // The default encoder is lossless. Avoid allocating and
            // zeroing the lossy DFS's 136 KiB of child metadata until a
            // lossy encode actually needs it.
            child_gen: Vec::new(),
            child_bits: Vec::new(),
        }
    }
}

struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl<'a> BitWriter<'a> {
    #[inline(always)]
    fn put(&mut self, code: u32, width: u32) {
        self.acc |= (code as u64) << self.nbits;
        self.nbits += width;
        // Drain four bytes at a time (codes are <= 12 bits, so the
        // accumulator never overflows 64 bits between drains); one 4-byte
        // extend beats four bounds-checked pushes.
        if self.nbits >= 32 {
            self.out.extend_from_slice(&(self.acc as u32).to_le_bytes());
            self.acc >>= 32;
            self.nbits -= 32;
        }
    }
    fn flush(&mut self) {
        while self.nbits > 0 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.nbits = self.nbits.saturating_sub(8);
        }
    }
}

impl LzwEncoder {
    fn ensure_lossy_scratch(&mut self) {
        if self.child_gen.is_empty() {
            self.child_gen.resize(MAX_CODE as usize, 0);
            self.child_bits.resize(MAX_CODE as usize * 4, 0);
        }
    }

    /// Reset the dictionary table (one L1-sized memset) and advance the
    /// generation stamp used by the lossy path's child bitmaps.
    #[inline(always)]
    fn bump_gen(&mut self) {
        self.table.fill(0);
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            // generation counter wrapped: reset the child stamps so stale
            // bitmaps can't hit (gen is never 0 afterwards)
            self.child_gen.iter_mut().for_each(|g| *g = 0);
            self.gen = 1;
        }
    }

    /// Probe for `key`: the code if present, else the slot where it
    /// should be inserted.
    #[inline(always)]
    fn probe(&self, key: u32) -> Result<u32, usize> {
        let mut slot = ((key.wrapping_mul(0x9E37_79B1)) >> (32 - TABLE_BITS)) as usize;
        loop {
            let e = self.table[slot];
            if e == 0 {
                return Err(slot); // empty
            }
            if (e >> 12) == key {
                return Ok(e & 0xFFF);
            }
            slot = (slot + 1) & (TABLE_SIZE - 1);
        }
    }

    /// Encode `data` (palette indices) and append the GIF image data section
    /// (min-code-size byte + length-prefixed sub-blocks + terminator) to `out`.
    /// With `lossy`, dictionary matches may continue through near-color
    /// substitutions within the map's error budget.
    /// `scale`, when given with `lossy`, is one byte per pixel of `data`
    /// (0..=255) scaling that pixel's error cap: the quantizer sets it
    /// below 255 in smooth undithered regions, where a substitution shows
    /// as a plateau or hatch rather than vanishing into dither.
    pub fn encode(
        &mut self,
        min_code_size: u8,
        data: &[u8],
        lossy: Option<&LossyMap>,
        scale: Option<&[u8]>,
        out: &mut Vec<u8>,
    ) {
        out.push(min_code_size);
        self.scratch.clear();
        let mut scratch = std::mem::take(&mut self.scratch);
        self.encode_raw(min_code_size, data, lossy, scale, &mut scratch);

        // Chunk into 255-byte sub-blocks.
        out.reserve(scratch.len() + scratch.len() / 255 + 2);
        for block in scratch.chunks(255) {
            out.push(block.len() as u8);
            out.extend_from_slice(block);
        }
        out.push(0); // block terminator
        self.scratch = scratch;
    }

    fn encode_raw(
        &mut self,
        min_code_size: u8,
        data: &[u8],
        lossy: Option<&LossyMap>,
        scale: Option<&[u8]>,
        out: &mut Vec<u8>,
    ) {
        if let Some(map) = lossy {
            return self.encode_raw_lossy(min_code_size, data, map, scale, out);
        }
        let clear = 1u32 << min_code_size;
        let eoi = clear + 1;
        let mut bw = BitWriter {
            out,
            acc: 0,
            nbits: 0,
        };
        let mut width = min_code_size as u32 + 1;
        bw.put(clear, width);
        if data.is_empty() {
            bw.put(eoi, width);
            bw.flush();
            return;
        }

        self.bump_gen();
        let mut next = eoi + 1;
        let mut cur = data[0] as u32;
        let mut run = 1u32;
        let mut run_ewma = 1u32 << RUN_EWMA_SCALE;

        for (i, &b) in data[1..].iter().enumerate() {
            match self.probe((cur << 8) | b as u32) {
                Ok(code) => {
                    cur = code;
                    run += 1;
                }
                Err(slot) => {
                    bw.put(cur, width);
                    update_run_ewma(&mut run_ewma, run);
                    run = 1;
                    if next < MAX_CODE {
                        let key = (cur << 8) | b as u32;
                        self.table[slot] = (key << 12) | next;
                        if next == (1 << width) {
                            width += 1;
                        }
                        next += 1;
                    } else if should_clear(run_ewma, data.len() - 1 - i, min_code_size as u32) {
                        bw.put(clear, width);
                        width = min_code_size as u32 + 1;
                        next = eoi + 1;
                        run_ewma = 1 << RUN_EWMA_SCALE;
                        self.bump_gen();
                    }
                    // else: dictionary frozen — keep matching against it
                    cur = b as u32;
                }
            }
        }
        bw.put(cur, width);
        // A decoder can only build the entry for a code once it has read
        // the code that follows, so it trails this loop by one entry for
        // the whole stream — and on the final code it catches up, adding
        // an entry that was never added here (the flush above emits `cur`
        // without extending the dictionary). When that entry is the one
        // that fills the dictionary, the decoder widens its codes before
        // reading the EOI. Widen to match, or a decoder that reads through
        // to the EOI runs off the end of the stream looking for it.
        if next == (1 << width) && width < MAX_WIDTH {
            width += 1;
        }
        bw.put(eoi, width);
        bw.flush();
    }

    /// Lossy encoding loop (gifsicle port): each iteration runs a DFS from
    /// the trie root to find the longest dictionary match whose per-pixel
    /// error stays under the cap, emits it, and defines one new entry with
    /// the actual (unsubstituted) next pixel.
    fn encode_raw_lossy(
        &mut self,
        min_code_size: u8,
        data: &[u8],
        map: &LossyMap,
        scale: Option<&[u8]>,
        out: &mut Vec<u8>,
    ) {
        debug_assert!(scale.is_none_or(|s| s.len() == data.len()));
        self.ensure_lossy_scratch();
        let clear = 1u32 << min_code_size;
        let eoi = clear + 1;
        let mut bw = BitWriter {
            out,
            acc: 0,
            nbits: 0,
        };
        let mut width = min_code_size as u32 + 1;
        bw.put(clear, width);
        if data.is_empty() {
            bw.put(eoi, width);
            bw.flush();
            return;
        }

        self.bump_gen();
        let mut next = eoi + 1;
        let mut gen = self.gen;
        let mut pos = 0usize;
        let mut run_ewma = 1u32 << RUN_EWMA_SCALE;

        while pos < data.len() {
            // First pixel of a match is always exact (gifsicle: the root
            // node is the actual symbol); substitutions apply from the
            // second pixel on.
            let mut ctx = DfsCtx {
                gen,
                data,
                map,
                scale,
                // Visit budget bounds the DFS on pathological data;
                // exhausting it degrades toward greedy matching, never to
                // wrong output.
                visits: 4096,
                best: LossyBest {
                    code: data[pos] as u32,
                    end: pos + 1,
                    diff: 0,
                },
            };
            self.lossy_dfs(&mut ctx, pos + 1, data[pos] as u32, [0; 3], 0);
            let best = ctx.best;
            bw.put(best.code, width);
            update_run_ewma(&mut run_ewma, (best.end - pos) as u32);
            pos = best.end;
            if pos >= data.len() {
                break;
            }
            if next < MAX_CODE {
                // Define (emitted code, actual next pixel). The entry can
                // already exist if the visit budget cut the DFS short.
                if let Err(slot) = self.probe((best.code << 8) | data[pos] as u32) {
                    let key = (best.code << 8) | data[pos] as u32;
                    self.table[slot] = (key << 12) | next;
                    // mirror the new child link into the DFS bitmap
                    let p = best.code as usize;
                    let sym = data[pos] as usize;
                    if self.child_gen[p] != gen {
                        self.child_gen[p] = gen;
                        self.child_bits[p * 4..p * 4 + 4].fill(0);
                    }
                    self.child_bits[p * 4 + (sym >> 6)] |= 1u64 << (sym & 63);
                    if next == (1 << width) {
                        width += 1;
                    }
                    next += 1;
                }
            } else if should_clear(run_ewma, data.len() - pos, min_code_size as u32) {
                bw.put(clear, width);
                width = min_code_size as u32 + 1;
                next = eoi + 1;
                run_ewma = 1 << RUN_EWMA_SCALE;
                self.bump_gen();
                gen = self.gen;
            }
            // else: dictionary frozen — keep matching against it
        }
        // Same trailing-EOI widening as the lossless loop above.
        if next == (1 << width) && width < MAX_WIDTH {
            width += 1;
        }
        bw.put(eoi, width);
        bw.flush();
    }

    /// Returns true when the search is finished for good: a zero-error
    /// match reaching the end of the data can't be beaten (nothing is
    /// longer, and equal length needs strictly lower error), so the whole
    /// DFS unwinds immediately.
    fn lossy_dfs(
        &self,
        ctx: &mut DfsCtx,
        pos: usize,
        node_code: u32,
        dither: [i32; 3],
        accum: u64,
    ) -> bool {
        let data = ctx.data;
        // Longest match wins; equal length prefers lower total error.
        if pos > ctx.best.end || (pos == ctx.best.end && accum < ctx.best.diff) {
            ctx.best = LossyBest {
                code: node_code,
                end: pos,
                diff: accum,
            };
            if pos >= data.len() && accum == 0 {
                return true;
            }
        }
        if pos >= data.len() || ctx.visits == 0 {
            return false;
        }
        ctx.visits -= 1;
        // Leaf cut and bitmap gate: skipping a symbol the node has no
        // child for is exactly what a missed hash probe would do, minus
        // the probe (and, for candidates, minus the diff computation).
        let p = node_code as usize;
        if self.child_gen[p] != ctx.gen {
            return false;
        }
        let cb = &self.child_bits[p * 4..p * 4 + 4];
        let b = data[pos];
        // Exact continuation: zero cost, dither decays.
        if cb[(b >> 6) as usize] & (1u64 << (b & 63)) != 0 {
            if let Ok(code) = self.probe((node_code << 8) | b as u32) {
                let nd = ctx.map.next_dither(b, b, &dither);
                if self.lossy_dfs(ctx, pos + 1, code, nd, accum) {
                    return true;
                }
            }
        }
        let map = ctx.map;
        if b != map.trans_idx {
            // per-pixel cap: the frame's loss scale, if any, applies here
            let cap = match ctx.scale {
                Some(s) => (map.max_diff * s[pos] as u32) >> 8,
                None => map.max_diff,
            };
            // Intersect this symbol's candidate set with the node's child
            // set before touching the list. Most nodes share no candidate
            // with their children at all, and of those that do, nearly all
            // share exactly one -- so the ordered scan below is reached
            // rarely, and 94% of its iterations (which were bit tests that
            // failed) disappear.
            let cm = map.cand_mask(b);
            let hits = [cm[0] & cb[0], cm[1] & cb[1], cm[2] & cb[2], cm[3] & cb[3]];
            let nhits = hits[0].count_ones()
                + hits[1].count_ones()
                + hits[2].count_ones()
                + hits[3].count_ones();
            if nhits == 1 {
                // Sole candidate child: no ordering to preserve, so the
                // distance-sorted scan (and its `cand_limit` early break)
                // has nothing left to decide. Dropping the limit test here
                // is not an approximation -- the limit provably only cuts
                // candidates that `d > cap` rejects anyway, so the explored
                // set is identical either way.
                let w = hits.iter().position(|&h| h != 0).unwrap();
                let b2 = (w * 64) as u8 | hits[w].trailing_zeros() as u8;
                let d = map.diff(b, b2, &dither);
                if d <= cap {
                    if let Ok(code) = self.probe((node_code << 8) | b2 as u32) {
                        let nd = map.next_dither(b, b2, &dither);
                        if self.lossy_dfs(ctx, pos + 1, code, nd, accum + d as u64) {
                            return true;
                        }
                    }
                }
            } else if nhits > 1 {
                // Several candidate children: fall back to the ordered scan,
                // which explores them in ascending-distance order. Equal-cost
                // matches are settled by exploration order (`ctx.best` takes
                // the strictly-better one), so that order has to hold.
                let limit = LossyMap::cand_limit(&dither, cap);
                for &packed in map.candidates(b) {
                    // Distance-sorted: the first entry over the limit means
                    // every entry after it is too.
                    if (packed >> 8) > limit {
                        break;
                    }
                    let b2 = packed as u8;
                    if hits[(b2 >> 6) as usize] & (1u64 << (b2 & 63)) == 0 {
                        continue;
                    }
                    let d = map.diff(b, b2, &dither);
                    if d > cap {
                        continue;
                    }
                    if let Ok(code) = self.probe((node_code << 8) | b2 as u32) {
                        let nd = map.next_dither(b, b2, &dither);
                        if self.lossy_dfs(ctx, pos + 1, code, nd, accum + d as u64) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Reference GIF-LZW decoder used by tests (and the integration test's
    /// full GIF decoder).
    pub fn lzw_decode(min_code_size: u8, mut bytes: &[u8], expect: usize) -> Vec<u8> {
        let clear = 1usize << min_code_size;
        let eoi = clear + 1;
        let mut dict: Vec<Vec<u8>> = Vec::new();
        let reset = |dict: &mut Vec<Vec<u8>>| {
            dict.clear();
            for i in 0..clear {
                dict.push(vec![i as u8]);
            }
            dict.push(vec![]); // clear
            dict.push(vec![]); // eoi
        };
        reset(&mut dict);
        let mut width = min_code_size as u32 + 1;
        let mut acc = 0u64;
        let mut nbits = 0u32;
        let mut out = Vec::with_capacity(expect);
        let mut prev: Option<usize> = None;
        loop {
            while nbits < width {
                let (&b, rest) = bytes.split_first().expect("ran out of lzw data");
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
                let p = &dict[prev.unwrap()];
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

    fn roundtrip(data: &[u8]) {
        roundtrip_at(8, data);
    }

    fn roundtrip_at(min_code_size: u8, data: &[u8]) {
        let mut enc = LzwEncoder::default();
        let mut raw = Vec::new();
        enc.encode_raw(min_code_size, data, None, None, &mut raw);
        let dec = lzw_decode(min_code_size, &raw, data.len());
        assert_eq!(dec, data, "roundtrip failed for len {}", data.len());
    }

    /// The trailing EOI must be as wide as the decoder expects it to be.
    /// A decoder builds the entry for code *k* only after reading code
    /// *k+1*, so it trails the encoder by one entry and then catches up on
    /// the final code — creating an entry the encoder never made. If that
    /// entry is the one that fills the dictionary, the decoder widens its
    /// codes while the encoder, which flushes its last code without adding
    /// anything, has no reason to. This band brackets the lengths whose
    /// final code lands exactly there; a flat run reaches the first
    /// boundary quickly at `min_code_size` 2.
    #[test]
    fn eoi_width_matches_decoder_at_dict_boundary() {
        for len in 50..80 {
            roundtrip_at(2, &vec![0u8; len]);
        }
    }

    /// The lossy loop flushes its last code the same way, so it needs the
    /// same widening. A lossy map that changes nothing keeps the code
    /// stream identical to the lossless one, so the same lengths land on
    /// the boundary.
    #[test]
    fn lossy_eoi_width_matches_decoder_at_dict_boundary() {
        // 4 colors so min_code_size is 2; level 0 admits no substitution
        let colors: Vec<[u8; 3]> = (0..4u16).map(|i| [(i * 64) as u8; 3]).collect();
        let map = LossyMap::build(&colors, 4, 0);
        for len in 50..80 {
            let data = vec![0u8; len];
            let mut enc = LzwEncoder::default();
            let mut raw = Vec::new();
            enc.encode_raw(2, &data, Some(&map), None, &mut raw);
            assert_eq!(lzw_decode(2, &raw, len), data, "lossy roundtrip len {len}");
        }
    }

    #[test]
    fn roundtrip_empty() {
        roundtrip(&[]);
    }

    #[test]
    fn roundtrip_small() {
        roundtrip(&[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]);
        roundtrip(&[0; 1000]);
        roundtrip(&[255; 3]);
    }

    #[test]
    fn roundtrip_forces_clears() {
        // pseudo-random data big enough to overflow the 4096-code dict many times
        let mut x = 12345u32;
        let data: Vec<u8> = (0..1_000_000)
            .map(|_| {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                (x >> 24) as u8
            })
            .collect();
        roundtrip(&data);
    }

    #[test]
    fn roundtrip_runs() {
        // long runs grow codes to max width without noise
        let mut data = Vec::new();
        for i in 0..64u32 {
            data.extend(std::iter::repeat_n((i % 7) as u8, 10_000));
        }
        roundtrip(&data);
    }

    #[test]
    fn encoder_reuse_across_frames() {
        let mut enc = LzwEncoder::default();
        for round in 0..5u32 {
            let data: Vec<u8> = (0..50_000)
                .map(|i| ((i as u32 * (round + 3)) % 251) as u8)
                .collect();
            let mut raw = Vec::new();
            enc.encode_raw(8, &data, None, None, &mut raw);
            assert_eq!(lzw_decode(8, &raw, data.len()), data);
        }
    }

    #[test]
    /// `cand_mask` is the membership bitmap the DFS intersects against a
    /// trie node's children, and it has to describe exactly the same set as
    /// `candidates()` -- a mask bit the list lacks would resurrect a
    /// substitution the radius/`LOSSY_MAX_CANDS` truncation dropped, and a
    /// missing bit would silently skip a legal one.
    #[test]
    fn cand_mask_matches_candidate_list() {
        // Two palettes with different candidate-set shapes: a dense ramp
        // (every symbol saturates LOSSY_MAX_CANDS) and a sparse spread
        // (most symbols have few or no candidates inside the radius).
        for step in [2u16, 40] {
            let colors: Vec<[u8; 3]> = (0..64u16)
                .map(|i| {
                    [
                        (i * step) as u8,
                        (i * step / 2) as u8,
                        255 - (i * step) as u8,
                    ]
                })
                .collect();
            let map = LossyMap::build(&colors, 64, 30);
            for sym in 0..=255usize {
                let mut from_list = [0u64; 4];
                for &packed in map.candidates(sym as u8) {
                    let j = packed as u8;
                    from_list[(j >> 6) as usize] |= 1u64 << (j & 63);
                }
                assert_eq!(
                    map.cand_mask(sym as u8),
                    from_list,
                    "step {step}, symbol {sym}"
                );
                // and the list never names a symbol twice, so popcount is len
                let bits: u32 = from_list.iter().map(|w| w.count_ones()).sum();
                assert_eq!(bits as usize, map.candidates(sym as u8).len());
            }
        }
    }

    fn lossy_shrinks_and_bounds_error() {
        // grayscale ramp palette: neighbors are 4 gray levels apart
        let colors: Vec<[u8; 3]> = (0..64u16).map(|i| [(i * 4) as u8; 3]).collect();
        let level = 30u32;
        let map = LossyMap::build(&colors, 64, level);
        // noisy data: base ramp + per-pixel jitter of +-1 index
        let mut x = 55555u32;
        let mut rng = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        let data: Vec<u8> = (0..200_000)
            .map(|i| {
                let base = (i / 512) % 60 + 2;
                (base + (rng() % 3) as i32 - 1) as u8
            })
            .collect();
        let mut enc = LzwEncoder::default();
        let mut exact = Vec::new();
        enc.encode_raw(8, &data, None, None, &mut exact);
        let mut lossy = Vec::new();
        enc.encode_raw(8, &data, Some(&map), None, &mut lossy);
        assert!(
            lossy.len() < exact.len() * 9 / 10,
            "lossy ({}) should be >10% smaller than exact ({})",
            lossy.len(),
            exact.len()
        );
        let dec = lzw_decode(8, &lossy, data.len());
        assert_eq!(dec.len(), data.len());
        // Substitutions only ever come from candidate lists built within
        // twice the cap radius, so raw per-pixel squared RGB error is
        // bounded by 4 * level * 10 ...
        let raw_cap = (level * 10 * 4) as i64;
        let mut sse = 0i64;
        for (i, (&got, &want)) in dec.iter().zip(&data).enumerate() {
            let d: i64 = (0..3)
                .map(|c| (colors[got as usize][c] as i64 - colors[want as usize][c] as i64).pow(2))
                .sum();
            assert!(d <= raw_cap, "pixel {i}: {want}->{got} d2 {d} > {raw_cap}");
            sse += d;
        }
        // ... and thanks to the error feedback, the mean squared error
        // stays around the per-pixel cap rather than the raw bound.
        let mse = sse / data.len() as i64;
        assert!(mse <= (level * 10) as i64, "mse {mse} exceeds cap");
    }

    #[test]
    fn lossy_never_touches_transparent_index() {
        let colors: Vec<[u8; 3]> = (0..8u16).map(|i| [(i * 32) as u8; 3]).collect();
        let trans = colors.len() as u8; // 8: not in colors, gets no candidates
        let map = LossyMap::build(&colors, trans, 200);
        let mut data = Vec::new();
        for i in 0..50_000 {
            data.push(if i % 7 < 3 { trans } else { (i % 8) as u8 });
        }
        let mut enc = LzwEncoder::default();
        let mut out = Vec::new();
        enc.encode_raw(8, &data, Some(&map), None, &mut out);
        let dec = lzw_decode(8, &out, data.len());
        for (i, (&got, &want)) in dec.iter().zip(&data).enumerate() {
            if want == trans || got == trans {
                assert_eq!(got, want, "transparent pixel rewritten at {i}");
            }
        }
    }

    #[test]
    fn subblock_framing() {
        let mut enc = LzwEncoder::default();
        let data = vec![7u8; 10_000];
        let mut out = Vec::new();
        enc.encode(8, &data, None, None, &mut out);
        assert_eq!(out[0], 8);
        // walk sub-blocks and collect payload
        let mut i = 1;
        let mut payload = Vec::new();
        loop {
            let n = out[i] as usize;
            i += 1;
            if n == 0 {
                break;
            }
            payload.extend_from_slice(&out[i..i + n]);
            i += n;
        }
        assert_eq!(i, out.len());
        assert_eq!(lzw_decode(8, &payload, data.len()), data);
    }
}
