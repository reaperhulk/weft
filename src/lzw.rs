//! GIF-flavor LZW encoder, tuned for throughput: open-addressed hash table
//! with generation stamps (no per-clear memset), 64-bit bit accumulator,
//! sub-block chunking done in a single trailing pass.

const MAX_CODE: u32 = 4096;
const TABLE_BITS: u32 = 15;
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
    cands: Vec<u8>,
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
            cands.extend(list.iter().map(|&(j, _)| j));
        }
        starts.push(cands.len() as u32);
        LossyMap {
            starts,
            cands,
            colors: ci,
            trans_idx,
            max_diff,
        }
    }

    #[inline(always)]
    fn candidates(&self, symbol: u8) -> &[u8] {
        let s = self.starts[symbol as usize] as usize;
        let e = self.starts[symbol as usize + 1] as usize;
        &self.cands[s..e]
    }

    /// gifsicle color_diff: squared error between wanted and written,
    /// taking the smaller of the fully-dithered and half-dithered views
    /// (dithering is opportunistic, not required).
    #[inline(always)]
    fn diff(&self, want: u8, got: u8, dither: &[i32; 3]) -> u32 {
        let a = &self.colors[want as usize];
        let b = &self.colors[got as usize];
        let mut dith = 0i64;
        let mut undith = 0i64;
        for c in 0..3 {
            let base = a[c] - b[c];
            dith += ((base + dither[c]) as i64).pow(2);
            undith += ((base + dither[c] / 2) as i64).pow(2);
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

struct LossyBest {
    code: u32,
    end: usize,
    diff: u64,
}

/// Reusable encoder state so per-frame allocations amortize away when a
/// thread encodes many frames.
pub struct LzwEncoder {
    // entry: [gen:16 | key:24 | code:16] packed into u64 (key is 12-bit
    // prefix code + 8-bit appended byte = 20 bits)
    table: Vec<u64>,
    gen: u16,
    scratch: Vec<u8>,
}

impl Default for LzwEncoder {
    fn default() -> Self {
        Self {
            table: vec![u64::MAX; TABLE_SIZE],
            gen: 0,
            scratch: Vec::new(),
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
    #[inline(always)]
    fn bump_gen(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            // generation counter wrapped: hard reset to avoid stale hits
            self.table.iter_mut().for_each(|e| *e = u64::MAX);
            self.gen = 1;
        }
    }

    /// Probe for `key` in the current generation: the code if present,
    /// else the slot where it should be inserted.
    #[inline(always)]
    fn probe(&self, gen: u16, key: u32) -> Result<u32, usize> {
        let mut slot = ((key.wrapping_mul(0x9E37_79B1)) >> (32 - TABLE_BITS)) as usize;
        loop {
            let e = self.table[slot];
            if (e >> 40) as u16 != gen {
                return Err(slot); // empty (stale generation)
            }
            if ((e >> 16) as u32 & 0xFF_FFFF) == key {
                return Ok((e & 0xFFFF) as u32);
            }
            slot = (slot + 1) & (TABLE_SIZE - 1);
        }
    }

    /// Encode `data` (palette indices) and append the GIF image data section
    /// (min-code-size byte + length-prefixed sub-blocks + terminator) to `out`.
    /// With `lossy`, dictionary matches may continue through near-color
    /// substitutions within the map's error budget.
    pub fn encode(
        &mut self,
        min_code_size: u8,
        data: &[u8],
        lossy: Option<&LossyMap>,
        out: &mut Vec<u8>,
    ) {
        out.push(min_code_size);
        self.scratch.clear();
        let mut scratch = std::mem::take(&mut self.scratch);
        self.encode_raw(min_code_size, data, lossy, &mut scratch);

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
        out: &mut Vec<u8>,
    ) {
        if let Some(map) = lossy {
            return self.encode_raw_lossy(min_code_size, data, map, out);
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
        let mut gen = self.gen;
        let mut run = 1u32;
        let mut run_ewma = 1u32 << RUN_EWMA_SCALE;

        for (i, &b) in data[1..].iter().enumerate() {
            match self.probe(gen, (cur << 8) | b as u32) {
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
                        self.table[slot] =
                            ((gen as u64) << 40) | ((key as u64) << 16) | next as u64;
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
                        gen = self.gen;
                    }
                    // else: dictionary frozen — keep matching against it
                    cur = b as u32;
                }
            }
        }
        bw.put(cur, width);
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
        out: &mut Vec<u8>,
    ) {
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
            let mut best = LossyBest {
                code: data[pos] as u32,
                end: pos + 1,
                diff: 0,
            };
            // Visit budget bounds the DFS on pathological data; exhausting
            // it degrades toward greedy matching, never to wrong output.
            let mut visits = 4096u32;
            self.lossy_dfs(
                gen,
                data,
                pos + 1,
                data[pos] as u32,
                [0; 3],
                0,
                &mut visits,
                &mut best,
                map,
            );
            bw.put(best.code, width);
            update_run_ewma(&mut run_ewma, (best.end - pos) as u32);
            pos = best.end;
            if pos >= data.len() {
                break;
            }
            if next < MAX_CODE {
                // Define (emitted code, actual next pixel). The entry can
                // already exist if the visit budget cut the DFS short.
                if let Err(slot) = self.probe(gen, (best.code << 8) | data[pos] as u32) {
                    let key = (best.code << 8) | data[pos] as u32;
                    self.table[slot] = ((gen as u64) << 40) | ((key as u64) << 16) | next as u64;
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
        bw.put(eoi, width);
        bw.flush();
    }

    #[allow(clippy::too_many_arguments)]
    fn lossy_dfs(
        &self,
        gen: u16,
        data: &[u8],
        pos: usize,
        node_code: u32,
        dither: [i32; 3],
        accum: u64,
        visits: &mut u32,
        best: &mut LossyBest,
        map: &LossyMap,
    ) {
        // Longest match wins; equal length prefers lower total error.
        if pos > best.end || (pos == best.end && accum < best.diff) {
            *best = LossyBest {
                code: node_code,
                end: pos,
                diff: accum,
            };
        }
        if pos >= data.len() || *visits == 0 {
            return;
        }
        *visits -= 1;
        let b = data[pos];
        // Exact continuation: zero cost, dither decays.
        if let Ok(code) = self.probe(gen, (node_code << 8) | b as u32) {
            let nd = map.next_dither(b, b, &dither);
            self.lossy_dfs(gen, data, pos + 1, code, nd, accum, visits, best, map);
        }
        if b != map.trans_idx {
            for &b2 in map.candidates(b) {
                let d = map.diff(b, b2, &dither);
                if d > map.max_diff {
                    continue;
                }
                if let Ok(code) = self.probe(gen, (node_code << 8) | b2 as u32) {
                    let nd = map.next_dither(b, b2, &dither);
                    self.lossy_dfs(
                        gen,
                        data,
                        pos + 1,
                        code,
                        nd,
                        accum + d as u64,
                        visits,
                        best,
                        map,
                    );
                }
            }
        }
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
        let mut enc = LzwEncoder::default();
        let mut raw = Vec::new();
        enc.encode_raw(8, data, None, &mut raw);
        let dec = lzw_decode(8, &raw, data.len());
        assert_eq!(dec, data, "roundtrip failed for len {}", data.len());
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
            enc.encode_raw(8, &data, None, &mut raw);
            assert_eq!(lzw_decode(8, &raw, data.len()), data);
        }
    }

    #[test]
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
        enc.encode_raw(8, &data, None, &mut exact);
        let mut lossy = Vec::new();
        enc.encode_raw(8, &data, Some(&map), &mut lossy);
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
        enc.encode_raw(8, &data, Some(&map), &mut out);
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
        enc.encode(8, &data, None, &mut out);
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
