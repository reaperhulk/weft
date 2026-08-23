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
    None,
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
    pub colors: &'a [[u8; 3]],
    pub nearest: &'a crate::palette::NearestMap,
    pub trans_idx: u8,
}

impl<'a> Quantizer<'a> {
    #[inline(always)]
    fn lookup(&self, r: u8, g: u8, b: u8) -> u8 {
        self.nearest.lookup(r, g, b)
    }

    /// Quantize an RGBA frame into palette indices. Returns true if any
    /// pixel was alpha-transparent.
    pub fn quantize(&self, rgba: &[u8], w: usize, h: usize, mode: Dither, out: &mut [u8]) -> bool {
        match mode {
            Dither::None => self.quantize_plain(rgba, out),
            Dither::Bayer => self.quantize_bayer(rgba, w, h, out),
            Dither::Sierra2_4a => self.quantize_diffuse(rgba, w, h, out, false),
            Dither::FloydSteinberg => self.quantize_diffuse(rgba, w, h, out, true),
        }
    }

    fn quantize_plain(&self, rgba: &[u8], out: &mut [u8]) -> bool {
        let mut has_alpha = false;
        for (px, o) in rgba.chunks_exact(4).zip(out.iter_mut()) {
            if px[3] < 128 {
                *o = self.trans_idx;
                has_alpha = true;
            } else {
                *o = self.lookup(px[0], px[1], px[2]);
            }
        }
        has_alpha
    }

    fn quantize_bayer(&self, rgba: &[u8], w: usize, h: usize, out: &mut [u8]) -> bool {
        let mut has_alpha = false;
        for y in 0..h {
            let row = &rgba[y * w * 4..(y + 1) * w * 4];
            let orow = &mut out[y * w..(y + 1) * w];
            let brow = &BAYER8[y & 7];
            for (x, (px, o)) in row.chunks_exact(4).zip(orow.iter_mut()).enumerate() {
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
                *o = self.lookup(r, g, b);
            }
        }
        has_alpha
    }

    fn quantize_diffuse(&self, rgba: &[u8], w: usize, h: usize, out: &mut [u8], fs: bool) -> bool {
        let mut has_alpha = false;
        // next-row error buffer with one pad cell on each side
        let mut next: Vec<[i32; 3]> = vec![[0; 3]; w + 2];
        let mut cur: Vec<[i32; 3]> = vec![[0; 3]; w + 2];
        for y in 0..h {
            std::mem::swap(&mut cur, &mut next);
            next.iter_mut().for_each(|e| *e = [0; 3]);
            let row = &rgba[y * w * 4..(y + 1) * w * 4];
            let orow = &mut out[y * w..(y + 1) * w];
            let mut carry = [0i32; 3]; // error flowing rightward within the row
            for (x, (px, o)) in row.chunks_exact(4).zip(orow.iter_mut()).enumerate() {
                if px[3] < 128 {
                    *o = self.trans_idx;
                    has_alpha = true;
                    carry = [0; 3];
                    continue;
                }
                let e = &cur[x + 1];
                let r = (px[0] as i32 + carry[0] + e[0]).clamp(0, 255);
                let g = (px[1] as i32 + carry[1] + e[1]).clamp(0, 255);
                let b = (px[2] as i32 + carry[2] + e[2]).clamp(0, 255);
                let idx = self.lookup(r as u8, g as u8, b as u8);
                *o = idx;
                let c = &self.colors[idx as usize];
                let er = r - c[0] as i32;
                let eg = g - c[1] as i32;
                let eb = b - c[2] as i32;
                if fs {
                    // Floyd–Steinberg: 7/16 right, 3/16 down-left, 5/16 down, 1/16 down-right
                    carry = [er * 7 / 16, eg * 7 / 16, eb * 7 / 16];
                    let dl = &mut next[x];
                    dl[0] += er * 3 / 16;
                    dl[1] += eg * 3 / 16;
                    dl[2] += eb * 3 / 16;
                    let d = &mut next[x + 1];
                    d[0] += er * 5 / 16;
                    d[1] += eg * 5 / 16;
                    d[2] += eb * 5 / 16;
                    let dr = &mut next[x + 2];
                    dr[0] += er / 16;
                    dr[1] += eg / 16;
                    dr[2] += eb / 16;
                } else {
                    // Sierra-2-4A: 2/4 right, 1/4 down-left, 1/4 down.
                    // Truncating division (like ffmpeg's `err*scale/(1<<n)`)
                    // — an arithmetic shift would round negative errors
                    // toward -inf and diffuse more than 100% of the error,
                    // which diverges into noise.
                    carry = [er / 2, eg / 2, eb / 2];
                    let dl = &mut next[x];
                    dl[0] += er / 4;
                    dl[1] += eg / 4;
                    dl[2] += eb / 4;
                    let d = &mut next[x + 1];
                    d[0] += er / 4;
                    d[1] += eg / 4;
                    d[2] += eb / 4;
                }
            }
        }
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
        let q = Quantizer { colors: &colors, nearest: &nm, trans_idx: 3 };
        let rgba = [255u8, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0];
        let mut out = [9u8; 4];
        let has_alpha = q.quantize(&rgba, 4, 1, Dither::None, &mut out);
        assert!(has_alpha);
        assert_eq!(out, [2, 0, 1, 3]);
    }

    #[test]
    fn diffuse_no_error_when_exact() {
        // With an exact palette, dithering must be a no-op.
        let colors = vec![[10u8, 20, 30], [200, 100, 50]];
        let nm = NearestMap::build(&colors);
        let q = Quantizer { colors: &colors, nearest: &nm, trans_idx: 2 };
        let mut rgba = Vec::new();
        for i in 0..64 {
            let c = &colors[i % 2];
            rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
        let mut out = vec![0u8; 64];
        q.quantize(&rgba, 8, 8, Dither::Sierra2_4a, &mut out);
        for (i, &o) in out.iter().enumerate() {
            assert_eq!(o as usize, i % 2);
        }
    }
}
