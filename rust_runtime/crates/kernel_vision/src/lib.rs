//! # kernel_vision
//!
//! Fused, vision-specific CPU kernels for use ahead of (or inside) a
//! `Runner` backend, targeting the two bottlenecks that generic ONNX/graph
//! optimizers (constant folding, Conv+BN fusion, ...) don't touch for
//! transformer-style vision backbones:
//!
//! - **`token_merge`**: ToMe-style ("Token Merging: Your ViT But Faster",
//!   Bolya et al. 2023) bipartite soft-matching token reduction. Merges the
//!   `r` most-similar token pairs between two halves of the sequence by
//!   averaging them, shrinking the token count (and therefore the cost of
//!   every subsequent transformer block) without retraining.
//! - **`windowed_attention`**: local/windowed self-attention computed with
//!   an online (streaming) softmax over key tiles, i.e. the core numerical
//!   trick from FlashAttention (Dao et al. 2022) applied to small 2D
//!   spatial windows instead of a full causal LM sequence — never
//!   materializes the full `N x N` score matrix, only `window x tile`.
//!
//! Both are plain CPU kernels (parallelized with `rayon`) meant to run
//! *before* export (as a `torch.nn.Module`-equivalent preprocessing step in
//! the Python client, see `oxidizedvision.kernels`) or as a Rust-side
//! post-processing/serving-time op — not ONNX ops themselves, since ONNX
//! has no native "merge tokens by similarity" primitive.

use ndarray::{s, Array1, Array2};
use rayon::prelude::*;

/// Merge the `r` most-similar token pairs in `tokens` (shape `[n, d]`),
/// returning a new `[n - r, d]` array.
///
/// Follows the ToMe bipartite soft-matching scheme: tokens are split into
/// two alternating partitions (even index -> partition A, odd index ->
/// partition B). For every token in A we find its most cosine-similar
/// partner in B (this step is embarrassingly parallel over A, done with
/// `rayon`). We then greedily take the `r` highest-similarity `(a, b)`
/// pairs (each token used at most once) and replace each pair with its
/// mean; the remaining, unmatched tokens pass through unchanged. Merged
/// tokens are appended after the passthrough tokens, matching the reference
/// ToMe implementation's output ordering.
///
/// # Panics
/// Panics if `r` is larger than the number of tokens in the smaller
/// partition (i.e. `r > n / 2`), since that many disjoint pairs can't
/// exist.
pub fn token_merge(tokens: &Array2<f32>, r: usize) -> Array2<f32> {
    let n = tokens.nrows();
    let d = tokens.ncols();
    if r == 0 || n < 2 {
        return tokens.clone();
    }

    let a_idx: Vec<usize> = (0..n).step_by(2).collect();
    let b_idx: Vec<usize> = (1..n).step_by(2).collect();
    assert!(
        r <= a_idx.len(),
        "token_merge: r={r} exceeds the number of source tokens available to merge ({})",
        a_idx.len()
    );

    let a_norm: Vec<Array1<f32>> = a_idx.iter().map(|&i| normalize(tokens.row(i))).collect();
    let b_norm: Vec<Array1<f32>> = b_idx.iter().map(|&i| normalize(tokens.row(i))).collect();

    // For each A token, best-matching B token by cosine similarity.
    // Parallel over A tokens: independent reductions, no shared mutable state.
    let best_matches: Vec<(usize, usize, f32)> = a_norm
        .par_iter()
        .enumerate()
        .map(|(ai, a_vec)| {
            let (bj, sim) = b_norm
                .iter()
                .enumerate()
                .map(|(bj, b_vec)| (bj, a_vec.dot(b_vec)))
                .fold((0usize, f32::NEG_INFINITY), |best, cur| {
                    if cur.1 > best.1 {
                        cur
                    } else {
                        best
                    }
                });
            (ai, bj, sim)
        })
        .collect();

    // Take the r highest-similarity (a, best-b) pairs. Unlike a bijective
    // matching, a single B token may absorb more than one A token here
    // (this is what the reference ToMe algorithm does too) — each merged A
    // token is removed and folded into its destination B token's running
    // average, so the output shrinks by exactly `r` regardless of how the
    // r chosen A tokens distribute over B destinations.
    let mut ranked = best_matches;
    ranked.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap());
    let merges = &ranked[..r];

    let mut used_a = vec![false; a_idx.len()];
    // (sum of merged-in values, count merged in) per B destination.
    let mut b_accum: Vec<Option<(Array1<f32>, usize)>> = vec![None; b_idx.len()];
    for &(ai, bj, _sim) in merges {
        used_a[ai] = true;
        let entry = b_accum[bj].get_or_insert_with(|| (Array1::zeros(d), 0));
        entry.0 += &tokens.row(a_idx[ai]);
        entry.1 += 1;
    }

    let mut out = Array2::<f32>::zeros((n - r, d));
    let mut row = 0usize;

    for (ai, &orig_i) in a_idx.iter().enumerate() {
        if !used_a[ai] {
            out.row_mut(row).assign(&tokens.row(orig_i));
            row += 1;
        }
    }
    for (bj, &orig_i) in b_idx.iter().enumerate() {
        let b_val = tokens.row(orig_i);
        match &b_accum[bj] {
            None => out.row_mut(row).assign(&b_val),
            Some((sum, count)) => {
                let merged = (&b_val.to_owned() + sum) / (*count as f32 + 1.0);
                out.row_mut(row).assign(&merged);
            }
        }
        row += 1;
    }

    out
}

fn normalize(v: ndarray::ArrayView1<f32>) -> Array1<f32> {
    let norm = v.dot(&v).sqrt().max(1e-12);
    v.to_owned() / norm
}

/// Self-attention over an `h x w` grid of `d`-dim tokens (row-major, so
/// token `(y, x)` lives at flat index `y * w + x`), restricted to
/// non-overlapping `window x window` spatial blocks, computed with an
/// online (streaming) softmax over key tiles of `key_tile` rows at a time.
///
/// This never materializes a full `[window*window, window*window]` score
/// matrix at once — only one `[window*window, key_tile]` tile — which is
/// the core memory-bandwidth trick behind FlashAttention, and is what makes
/// this scale to larger windows without quadratic scratch memory. It is
/// mathematically exact softmax attention restricted to each window (not
/// an approximation), just computed tile-by-tile.
///
/// `x` has shape `[h * w, d]`. `h` and `w` must both be divisible by
/// `window`. Returns a `[h * w, d]` array of attention outputs, one window
/// at a time, independent across windows (parallelized with `rayon`).
pub fn windowed_attention(
    x: &Array2<f32>,
    h: usize,
    w: usize,
    window: usize,
    key_tile: usize,
) -> Array2<f32> {
    let d = x.ncols();
    assert_eq!(x.nrows(), h * w, "windowed_attention: x rows must equal h*w");
    assert!(window > 0 && key_tile > 0);
    assert_eq!(h % window, 0, "h must be divisible by window");
    assert_eq!(w % window, 0, "w must be divisible by window");

    let scale = 1.0 / (d as f32).sqrt();
    let wins_y = h / window;
    let wins_x = w / window;
    let n_windows = wins_y * wins_x;

    // Gather per-window token blocks first (cheap copy), then run the
    // online-softmax attention independently per window in parallel.
    let window_tokens: Vec<(Vec<usize>, Array2<f32>)> = (0..n_windows)
        .map(|wi| {
            let wy = wi / wins_x;
            let wx = wi % wins_x;
            let mut flat_idx = Vec::with_capacity(window * window);
            for dy in 0..window {
                for dx in 0..window {
                    let y = wy * window + dy;
                    let xcol = wx * window + dx;
                    flat_idx.push(y * w + xcol);
                }
            }
            let block = Array2::from_shape_fn((flat_idx.len(), d), |(i, j)| x[[flat_idx[i], j]]);
            (flat_idx, block)
        })
        .collect();

    let mut out = Array2::<f32>::zeros((h * w, d));
    let results: Vec<(Vec<usize>, Array2<f32>)> = window_tokens
        .into_par_iter()
        .map(|(flat_idx, block)| {
            let out_block = flash_softmax_attention(&block, &block, &block, scale, key_tile);
            (flat_idx, out_block)
        })
        .collect();

    for (flat_idx, out_block) in results {
        for (row, &idx) in flat_idx.iter().enumerate() {
            out.row_mut(idx).assign(&out_block.row(row));
        }
    }
    out
}

/// Single-head scaled-dot-product attention `softmax(Q K^T * scale) V`,
/// computed with running-max/running-sum online softmax over `key_tile`-row
/// chunks of `k`/`v` (the FlashAttention algorithm), so peak scratch memory
/// is `[q.nrows(), key_tile]` instead of `[q.nrows(), k.nrows()]`.
fn flash_softmax_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    scale: f32,
    key_tile: usize,
) -> Array2<f32> {
    let nq = q.nrows();
    let nk = k.nrows();
    let d = v.ncols();

    let mut acc = Array2::<f32>::zeros((nq, d));
    let mut row_max = Array1::<f32>::from_elem(nq, f32::NEG_INFINITY);
    let mut row_sum = Array1::<f32>::zeros(nq);

    let mut start = 0usize;
    while start < nk {
        let end = (start + key_tile).min(nk);
        let k_tile = k.slice(s![start..end, ..]);
        let v_tile = v.slice(s![start..end, ..]);

        // scores: [nq, tile]
        let scores = q.dot(&k_tile.t()).mapv(|s| s * scale);

        for i in 0..nq {
            let tile_max = scores
                .row(i)
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let new_max = row_max[i].max(tile_max);

            let correction = if row_max[i] == f32::NEG_INFINITY {
                0.0
            } else {
                (row_max[i] - new_max).exp()
            };
            row_sum[i] *= correction;
            for j in 0..d {
                acc[[i, j]] *= correction;
            }

            for (t, &s) in scores.row(i).iter().enumerate() {
                let p = (s - new_max).exp();
                row_sum[i] += p;
                for j in 0..d {
                    acc[[i, j]] += p * v_tile[[t, j]];
                }
            }
            row_max[i] = new_max;
        }
        start = end;
    }

    for i in 0..nq {
        let denom = row_sum[i].max(1e-12);
        for j in 0..d {
            acc[[i, j]] /= denom;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn token_merge_reduces_row_count() {
        let tokens = Array2::from_shape_fn((8, 4), |(i, j)| (i * 4 + j) as f32);
        let merged = token_merge(&tokens, 2);
        assert_eq!(merged.shape(), &[6, 4]);
    }

    #[test]
    fn token_merge_zero_r_is_identity() {
        let tokens = Array2::from_shape_fn((6, 3), |(i, j)| (i + j) as f32);
        let merged = token_merge(&tokens, 0);
        assert_eq!(merged, tokens);
    }

    #[test]
    fn token_merge_merges_identical_duplicates_first() {
        // Token 0 and 1 are identical (cos sim = 1) -> should be merged into itself.
        let tokens = array![
            [1.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let merged = token_merge(&tokens, 1);
        assert_eq!(merged.shape(), &[3, 3]);
        // Row 1 is where token 1 (B partition, index 0) lands; it absorbed
        // the identical token 0, so it should be unchanged: [1, 0, 0].
        let merged_row = merged.row(1);
        assert!((merged_row[0] - 1.0).abs() < 1e-6);
        assert!(merged_row[1].abs() < 1e-6);
    }

    #[test]
    #[should_panic]
    fn token_merge_panics_on_too_large_r() {
        let tokens = Array2::<f32>::zeros((4, 2));
        let _ = token_merge(&tokens, 10);
    }

    #[test]
    fn windowed_attention_matches_full_attention_within_single_window() {
        // When window == grid size, windowed_attention must equal exact
        // full self-attention over the whole sequence.
        let h = 2;
        let w = 2;
        let d = 3;
        let x = Array2::from_shape_fn((h * w, d), |(i, j)| ((i + 1) as f32 * (j + 1) as f32) * 0.1);

        let got = windowed_attention(&x, h, w, 2, 1);
        let want = flash_softmax_attention(&x, &x, &x, 1.0 / (d as f32).sqrt(), 100);

        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn windowed_attention_key_tile_size_does_not_change_result() {
        let h = 4;
        let w = 4;
        let d = 5;
        let x = Array2::from_shape_fn((h * w, d), |(i, j)| ((i * 7 + j * 3) % 11) as f32 * 0.07);

        let tile1 = windowed_attention(&x, h, w, 2, 1);
        let tile_big = windowed_attention(&x, h, w, 2, 100);

        for (a, b) in tile1.iter().zip(tile_big.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn windowed_attention_output_shape() {
        let h = 4;
        let w = 6;
        let d = 8;
        let x = Array2::<f32>::zeros((h * w, d));
        let out = windowed_attention(&x, h, w, 2, 4);
        assert_eq!(out.shape(), &[h * w, d]);
    }
}
