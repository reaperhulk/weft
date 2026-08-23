//! sRGB <-> OkLab, matching the transform ffmpeg's palettegen/paletteuse
//! use (theirs is fixed-point; f32 here — palette selection and nearest
//! matching don't need bit-exactness, just the same geometry).

use std::sync::OnceLock;

fn linear_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            let x = i as f32 / 255.0;
            *v = if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            };
        }
        t
    })
}

pub struct LabConverter {
    lut: &'static [f32; 256],
}

impl LabConverter {
    pub fn new() -> Self {
        LabConverter { lut: linear_lut() }
    }

    #[inline(always)]
    pub fn srgb_to_oklab(&self, r: u8, g: u8, b: u8) -> [f32; 3] {
        let lr = self.lut[r as usize];
        let lg = self.lut[g as usize];
        let lb = self.lut[b as usize];
        let l = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb;
        let m = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb;
        let s = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb;
        let l_ = l.cbrt();
        let m_ = m.cbrt();
        let s_ = s.cbrt();
        [
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ]
    }
}

pub fn oklab_to_srgb(lab: [f32; 3]) -> [u8; 3] {
    let l_ = lab[0] + 0.3963377774 * lab[1] + 0.2158037573 * lab[2];
    let m_ = lab[0] - 0.1055613458 * lab[1] - 0.0638541728 * lab[2];
    let s_ = lab[0] - 0.0894841775 * lab[1] - 1.2914855480 * lab[2];
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    let r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    let g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    let b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
    [linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b)]
}

fn linear_to_srgb(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let v = if x <= 0.0031308 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (v * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_extremes() {
        let cv = LabConverter::new();
        for c in [
            [0u8, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [17, 130, 200],
        ] {
            let lab = cv.srgb_to_oklab(c[0], c[1], c[2]);
            let back = oklab_to_srgb(lab);
            for i in 0..3 {
                assert!(
                    (back[i] as i32 - c[i] as i32).abs() <= 1,
                    "{c:?} -> {back:?}"
                );
            }
        }
    }

    #[test]
    fn white_l_is_one() {
        let cv = LabConverter::new();
        let lab = cv.srgb_to_oklab(255, 255, 255);
        assert!((lab[0] - 1.0).abs() < 1e-3);
        assert!(lab[1].abs() < 1e-3 && lab[2].abs() < 1e-3);
    }
}
