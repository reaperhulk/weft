//! SIMD kernels (fearless_simd): runtime-dispatched, so the shipped
//! baseline-CPU binaries still use AVX2/NEON where the machine has it.

use fearless_simd::{f32x8, Level, Simd, SimdBase};

/// Palette OkLab channels in structure-of-arrays form, padded to a
/// multiple of 8 lanes with +inf (padding never wins a min and never
/// passes a <= bound test).
pub struct PalSoa {
    pub l: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl PalSoa {
    pub fn new(labs: &[[f32; 3]]) -> Self {
        let padded = labs.len().div_ceil(8) * 8;
        let mut l = vec![f32::INFINITY; padded];
        let mut a = vec![f32::INFINITY; padded];
        let mut b = vec![f32::INFINITY; padded];
        for (i, lab) in labs.iter().enumerate() {
            l[i] = lab[0];
            a[i] = lab[1];
            b[i] = lab[2];
        }
        PalSoa { l, a, b }
    }
}

/// Fill `dists[i]` with the squared OkLab distance from `q` to palette
/// color i and return the minimum. `dists` must be at least the padded
/// palette length.
pub fn cell_distances(level: Level, pal: &PalSoa, q: [f32; 3], dists: &mut [f32]) -> f32 {
    fearless_simd::dispatch!(level, simd => cell_distances_impl(simd, pal, q, dists))
}

#[inline(always)]
fn cell_distances_impl<S: Simd>(simd: S, pal: &PalSoa, q: [f32; 3], dists: &mut [f32]) -> f32 {
    let ql = f32x8::splat(simd, q[0]);
    let qa = f32x8::splat(simd, q[1]);
    let qb = f32x8::splat(simd, q[2]);
    let mut minv = f32x8::splat(simd, f32::MAX);
    let n = pal.l.len();
    let mut i = 0;
    while i < n {
        let dl = f32x8::from_slice(simd, &pal.l[i..i + 8]) - ql;
        let da = f32x8::from_slice(simd, &pal.a[i..i + 8]) - qa;
        let db = f32x8::from_slice(simd, &pal.b[i..i + 8]) - qb;
        let d = dl * dl + da * da + db * db;
        d.store_slice(&mut dists[i..i + 8]);
        minv = minv.min(d);
        i += 8;
    }
    let arr: [f32; 8] = minv.into();
    arr.iter().fold(f32::MAX, |m, &v| m.min(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_scalar() {
        let labs: Vec<[f32; 3]> = (0..255)
            .map(|i| {
                let f = i as f32;
                [f * 0.001, (f * 0.7).sin() * 0.2, (f * 0.3).cos() * 0.2]
            })
            .collect();
        let pal = PalSoa::new(&labs);
        let level = Level::new();
        let q = [0.4f32, -0.05, 0.11];
        let mut dists = vec![0f32; pal.l.len()];
        let dmin = cell_distances(level, &pal, q, &mut dists);
        let mut smin = f32::MAX;
        for (i, lab) in labs.iter().enumerate() {
            let d = (lab[0] - q[0]).powi(2) + (lab[1] - q[1]).powi(2) + (lab[2] - q[2]).powi(2);
            assert!((dists[i] - d).abs() <= d * 1e-6 + 1e-12, "idx {i}");
            smin = smin.min(d);
        }
        assert!((dmin - smin).abs() <= smin * 1e-6 + 1e-12);
        // padding lanes must never look attractive
        for &d in &dists[labs.len()..] {
            assert!(d.is_infinite());
        }
    }
}
