//! GIF-flavor LZW encoder, tuned for throughput: open-addressed hash table
//! with generation stamps (no per-clear memset), 64-bit bit accumulator,
//! sub-block chunking done in a single trailing pass.

const MAX_CODE: u32 = 4096;
const TABLE_BITS: u32 = 15;
const TABLE_SIZE: usize = 1 << TABLE_BITS;

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
        while self.nbits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }
    fn flush(&mut self) {
        if self.nbits > 0 {
            self.out.push(self.acc as u8);
            self.acc = 0;
            self.nbits = 0;
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

    /// Encode `data` (palette indices) and append the GIF image data section
    /// (min-code-size byte + length-prefixed sub-blocks + terminator) to `out`.
    pub fn encode(&mut self, min_code_size: u8, data: &[u8], out: &mut Vec<u8>) {
        out.push(min_code_size);
        self.scratch.clear();
        let mut scratch = std::mem::take(&mut self.scratch);
        self.encode_raw(min_code_size, data, &mut scratch);

        // Chunk into 255-byte sub-blocks.
        out.reserve(scratch.len() + scratch.len() / 255 + 2);
        for block in scratch.chunks(255) {
            out.push(block.len() as u8);
            out.extend_from_slice(block);
        }
        out.push(0); // block terminator
        self.scratch = scratch;
    }

    fn encode_raw(&mut self, min_code_size: u8, data: &[u8], out: &mut Vec<u8>) {
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
        let gen_tag = (self.gen as u64) << 40;

        let mut gen_tag = gen_tag;
        for &b in &data[1..] {
            let key = (cur << 8) | b as u32;
            // multiplicative hash; linear probe
            let mut slot = ((key.wrapping_mul(0x9E37_79B1)) >> (32 - TABLE_BITS)) as usize;
            let found = loop {
                let e = self.table[slot];
                if (e >> 40) != (gen_tag >> 40) {
                    break None; // empty (stale generation)
                }
                if ((e >> 16) as u32 & 0xFF_FFFF) == key {
                    break Some((e & 0xFFFF) as u32);
                }
                slot = (slot + 1) & (TABLE_SIZE - 1);
            };
            match found {
                Some(code) => cur = code,
                None => {
                    bw.put(cur, width);
                    if next < MAX_CODE {
                        self.table[slot] = gen_tag | ((key as u64) << 16) | next as u64;
                        if next == (1 << width) {
                            width += 1;
                        }
                        next += 1;
                    } else {
                        bw.put(clear, width);
                        width = min_code_size as u32 + 1;
                        next = eoi + 1;
                        self.bump_gen();
                        gen_tag = (self.gen as u64) << 40;
                    }
                    cur = b as u32;
                }
            }
        }
        bw.put(cur, width);
        bw.put(eoi, width);
        bw.flush();
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
        enc.encode_raw(8, data, &mut raw);
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
            enc.encode_raw(8, &data, &mut raw);
            assert_eq!(lzw_decode(8, &raw, data.len()), data);
        }
    }

    #[test]
    fn subblock_framing() {
        let mut enc = LzwEncoder::default();
        let data = vec![7u8; 10_000];
        let mut out = Vec::new();
        enc.encode(8, &data, &mut out);
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
