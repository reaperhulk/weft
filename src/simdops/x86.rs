//! Kernels for the AVX-512 subset available before Ice Lake. The generic
//! SIMD backend requires Ice Lake features that Cascade Lake does not have.
use std::arch::x86_64::*;

// Short palettes do too little work to amortize the wider kernels.
// Four vectors was the best cutoff in the Cascade Lake clip benchmark.
pub(super) const MIN_PALETTE_LEN: usize = 64;

pub(super) fn has_avx512() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
}

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn cell_distances(pal: &super::PalSoa, q: [f32; 3], dists: &mut [f32]) -> f32 {
    let n = pal.l.len();
    assert!(n.is_multiple_of(16));
    assert!(pal.a.len() >= n && pal.b.len() >= n && dists.len() >= n);
    let ql = _mm512_set1_ps(q[0]);
    let qa = _mm512_set1_ps(q[1]);
    let qb = _mm512_set1_ps(q[2]);
    let mut minv = _mm512_set1_ps(f32::MAX);
    for i in (0..n).step_by(16) {
        let dl = _mm512_sub_ps(_mm512_loadu_ps(pal.l.as_ptr().add(i)), ql);
        let da = _mm512_sub_ps(_mm512_loadu_ps(pal.a.as_ptr().add(i)), qa);
        let db = _mm512_sub_ps(_mm512_loadu_ps(pal.b.as_ptr().add(i)), qb);
        // Keep the generic backend's multiply/add order; contracting to FMA
        // can change palette assignments near equal-distance boundaries.
        let d = _mm512_add_ps(
            _mm512_add_ps(_mm512_mul_ps(dl, dl), _mm512_mul_ps(da, da)),
            _mm512_mul_ps(db, db),
        );
        _mm512_storeu_ps(dists.as_mut_ptr().add(i), d);
        minv = _mm512_min_ps(minv, d);
    }
    _mm512_reduce_min_ps(minv)
}

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn cell_candidates(dists: &[f32], bound2: f32, arena: &mut Vec<u8>) {
    let bound = _mm512_set1_ps(bound2);
    let end = dists.len() / 16 * 16;
    for i in (0..end).step_by(16) {
        let d = _mm512_loadu_ps(dists.as_ptr().add(i));
        let mut mask = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(d, bound);
        while mask != 0 {
            arena.push((i + mask.trailing_zeros() as usize) as u8);
            mask &= mask - 1;
        }
    }
    for (i, &d) in dists.iter().enumerate().skip(end) {
        if d <= bound2 {
            arena.push(i as u8);
        }
    }
}

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn nearest_color(pal: &super::PalSoa, q: [f32; 3]) -> usize {
    let n = pal.l.len();
    assert!(n.is_multiple_of(16));
    assert!(pal.a.len() >= n && pal.b.len() >= n);
    let ql = _mm512_set1_ps(q[0]);
    let qa = _mm512_set1_ps(q[1]);
    let qb = _mm512_set1_ps(q[2]);
    let mut minv = _mm512_set1_ps(f32::MAX);
    let mut indices = _mm512_set1_epi32(i32::MAX);
    let mut current = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    for i in (0..n).step_by(16) {
        let dl = _mm512_sub_ps(_mm512_loadu_ps(pal.l.as_ptr().add(i)), ql);
        let da = _mm512_sub_ps(_mm512_loadu_ps(pal.a.as_ptr().add(i)), qa);
        let db = _mm512_sub_ps(_mm512_loadu_ps(pal.b.as_ptr().add(i)), qb);
        let d = _mm512_add_ps(
            _mm512_add_ps(_mm512_mul_ps(dl, dl), _mm512_mul_ps(da, da)),
            _mm512_mul_ps(db, db),
        );
        // Strict improvement keeps the first index within each lane.
        let better = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(d, minv);
        indices = _mm512_mask_blend_epi32(better, indices, current);
        minv = _mm512_min_ps(minv, d);
        current = _mm512_add_epi32(current, _mm512_set1_epi32(16));
    }
    let min = _mm512_reduce_min_ps(minv);
    let matches = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(minv, _mm512_set1_ps(min));
    // Equal minima in different lanes must also prefer the first palette
    // index, regardless of the lane in which it occurred.
    let index = _mm512_mask_reduce_min_epi32(matches, indices);
    if index == i32::MAX {
        0
    } else {
        index as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(seed: &mut u32) -> u8 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*seed >> 24) as u8
    }

    #[test]
    fn palette_distances_match_generic_backend() {
        if !has_avx512() {
            return;
        }
        let mut seed = 91;
        for n in [0, 1, 15, 16, 17, 31, 127, 254, 255] {
            let labs: Vec<[f32; 3]> = (0..n)
                .map(|_| {
                    [
                        random(&mut seed) as f32 / 255.0,
                        random(&mut seed) as f32 / 255.0 - 0.5,
                        random(&mut seed) as f32 / 255.0 - 0.5,
                    ]
                })
                .collect();
            let pal = super::super::PalSoa::new(&labs);
            for _ in 0..30 {
                let q = [
                    random(&mut seed) as f32 / 255.0,
                    random(&mut seed) as f32 / 255.0 - 0.5,
                    random(&mut seed) as f32 / 255.0 - 0.5,
                ];
                let mut expected = vec![0.0; pal.l.len()];
                let level = super::super::level();
                let min = fearless_simd::dispatch!(level, simd => super::super::cell_distances_impl(simd, &pal, q, &mut expected));
                let mut got = vec![-1.0; pal.l.len() + 2];
                // SAFETY: Feature detection above; output includes the padded palette.
                let actual_min = unsafe { cell_distances(&pal, q, &mut got[1..pal.l.len() + 1]) };
                assert_eq!(actual_min, min);
                let nearest = expected[..n].iter().position(|&d| d == min).unwrap_or(0);
                // SAFETY: Feature detection above; PalSoa has padded channels.
                assert_eq!(unsafe { nearest_color(&pal, q) }, nearest);
                assert_eq!(&got[1..pal.l.len() + 1], expected);
                assert_eq!((got[0], got[pal.l.len() + 1]), (-1.0, -1.0));
            }
        }
    }

    #[test]
    fn nearest_color_breaks_ties_by_palette_index() {
        if !has_avx512() {
            return;
        }
        for (first, second) in [(0, 16), (15, 16), (17, 32), (31, 240), (239, 254)] {
            let mut labs = vec![[1.0, 1.0, 1.0]; 255];
            labs[first] = [0.0, 0.0, 0.0];
            labs[second] = [0.0, 0.0, 0.0];
            let pal = super::super::PalSoa::new(&labs);
            // SAFETY: Feature detection above; PalSoa has padded channels.
            assert_eq!(unsafe { nearest_color(&pal, [0.0; 3]) }, first);
        }
    }

    #[test]
    fn candidates_preserve_order_boundaries_and_prefix() {
        if !has_avx512() {
            return;
        }
        for n in 0..=255 {
            let dists: Vec<f32> = (0..n)
                .map(|i| [0.0, -0.0, 1.0, 2.0, f32::INFINITY, f32::NAN, f32::MAX][i % 7])
                .collect();
            for bound in [-1.0, 0.0, 1.0, 2.0, f32::MAX, f32::INFINITY, f32::NAN] {
                let mut expected = vec![31, 23];
                expected.extend(
                    dists
                        .iter()
                        .enumerate()
                        .filter(|(_, d)| **d <= bound)
                        .map(|(i, _)| i as u8),
                );
                let mut got = vec![31, 23];
                // SAFETY: Feature detection above; the kernel reads full vectors only.
                unsafe {
                    cell_candidates(&dists, bound, &mut got);
                }
                assert_eq!(got, expected, "n={n}, bound={bound}");
            }
        }
    }
}
