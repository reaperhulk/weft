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

    /// Linearized sRGB channel value (the LUT the Lab transform uses).
    #[inline(always)]
    pub fn linear(&self, v: u8) -> f32 {
        self.lut[v as usize]
    }

    #[inline(always)]
    pub fn srgb_to_oklab(&self, r: u8, g: u8, b: u8) -> [f32; 3] {
        self.srgb_to_oklab_with(r, g, b, f32::cbrt)
    }

    /// `srgb_to_oklab` with the fast cube root — for per-pixel hot paths.
    #[inline(always)]
    pub fn srgb_to_oklab_fast(&self, r: u8, g: u8, b: u8) -> [f32; 3] {
        let [l, m, s] = self.lms(r, g, b);
        let [l_, m_, s_] = cbrt3_fast(l, m, s);
        lab_from_cbrt_lms(l_, m_, s_)
    }

    #[inline(always)]
    fn srgb_to_oklab_with(&self, r: u8, g: u8, b: u8, cbrt: impl Fn(f32) -> f32) -> [f32; 3] {
        let [l, m, s] = self.lms(r, g, b);
        lab_from_cbrt_lms(cbrt(l), cbrt(m), cbrt(s))
    }

    /// Linear-light LMS cone response for an sRGB colour.
    #[inline(always)]
    fn lms(&self, r: u8, g: u8, b: u8) -> [f32; 3] {
        let lr = self.lut[r as usize];
        let lg = self.lut[g as usize];
        let lb = self.lut[b as usize];
        [
            0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb,
            0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb,
            0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb,
        ]
    }
}

/// OkLab from the cube-rooted LMS response.
#[inline(always)]
fn lab_from_cbrt_lms(l_: f32, m_: f32, s_: f32) -> [f32; 3] {
    [
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ]
}

/// Fast cube root for non-negative finite f32: Kahan-style bit-trick seed
/// plus two Halley iterations (cubic convergence), landing within a couple
/// of ULPs of libm's cbrtf at a fraction of the cost. Used on hot paths
/// where the OkLab *geometry* matters but correctly-rounded rounding does
/// not (nearest-color argmins; candidate bounds carry an explicit margin).
///
/// On x86_64 the hot path uses the packed `cbrt3_fast`; this scalar form
/// is its reference (the test proves them bit-identical) and the path
/// other architectures take.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[inline(always)]
pub fn cbrt_fast(x: f32) -> f32 {
    let mut y = f32::from_bits(cbrt_seed_bits(x));
    // Halley: y <- y * (y^3 + 2x) / (2y^3 + x); at x == 0 this decays y
    // toward zero, so no special case is needed.
    let y3 = y * y * y;
    y *= (y3 + 2.0 * x) / (2.0 * y3 + x);
    let y3 = y * y * y;
    y *= (y3 + 2.0 * x) / (2.0 * y3 + x);
    y
}

#[inline(always)]
fn cbrt_seed_bits(x: f32) -> u32 {
    x.to_bits() / 3 + 709_921_077
}

/// Three `cbrt_fast`s in one 4-lane vector, bit-identical to calling
/// `cbrt_fast` per channel (same operations in the same order, IEEE
/// per lane, no contraction).
///
/// Written out explicitly rather than left to the compiler for a
/// reason: built with AVX enabled (`-C target-cpu=x86-64-v3`), LLVM's
/// SLP vectoriser packs the three scalar chains into one 128-bit vector
/// itself -- but pads the fourth lane with whatever it has, which is a
/// zero from the LMS matrix. The seed for x = 0 cubes to a subnormal,
/// and every subnormal operand costs a ~150-cycle microcode assist on
/// Intel cores, on every call: `resolve_off` went from 6% to 24% of all
/// cycles and the "v3 build is 25% slower" that Cargo.toml used to warn
/// about was entirely this. The pad lane here is 1.0, whose seed is
/// normal, so the packed form is safe under any target features.
#[inline(always)]
pub fn cbrt3_fast(l: f32, m: f32, s: f32) -> [f32; 3] {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE2 is part of the x86_64 baseline, so these intrinsics
    // are always available; the loads and stores are on local arrays.
    unsafe {
        use std::arch::x86_64::*;
        let x = _mm_set_ps(1.0, s, m, l);
        let mut y = _mm_castsi128_ps(_mm_set_epi32(
            cbrt_seed_bits(1.0) as i32,
            cbrt_seed_bits(s) as i32,
            cbrt_seed_bits(m) as i32,
            cbrt_seed_bits(l) as i32,
        ));
        let x2 = _mm_add_ps(x, x);
        for _ in 0..2 {
            let y3 = _mm_mul_ps(_mm_mul_ps(y, y), y);
            let num = _mm_add_ps(y3, x2);
            let den = _mm_add_ps(_mm_add_ps(y3, y3), x);
            y = _mm_mul_ps(y, _mm_div_ps(num, den));
        }
        let mut out = [0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), y);
        [out[0], out[1], out[2]]
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        [cbrt_fast(l), cbrt_fast(m), cbrt_fast(s)]
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
    fn cbrt3_fast_matches_scalar_on_every_colour() {
        // Every sRGB colour, so the packed path is proven bit-identical
        // on the exact inputs the nearest-colour scan feeds it.
        let cv = LabConverter::new();
        let step = if cfg!(debug_assertions) { 7 } else { 1 };
        for c in (0..1u32 << 24).step_by(step) {
            let (r, g, b) = ((c >> 16) as u8, (c >> 8) as u8, c as u8);
            let want = cv.srgb_to_oklab_with(r, g, b, cbrt_fast);
            let got = cv.srgb_to_oklab_fast(r, g, b);
            assert_eq!(
                want.map(f32::to_bits),
                got.map(f32::to_bits),
                "({r},{g},{b}): {want:?} vs {got:?}"
            );
        }
    }

    #[test]
    fn cbrt_fast_accuracy() {
        // seed quality + two Halley steps must land within a few ULPs
        for i in 0..=100_000u32 {
            let x = i as f32 / 100_000.0;
            let got = cbrt_fast(x);
            let want = x.cbrt();
            assert!(
                (got - want).abs() <= want * 1e-6 + 1e-9,
                "cbrt_fast({x}) = {got}, libm {want}"
            );
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
