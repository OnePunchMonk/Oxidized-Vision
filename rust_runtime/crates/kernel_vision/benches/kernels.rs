use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kernel_vision::{token_merge, windowed_attention};
use ndarray::Array2;

/// Naive O(n^2) full self-attention, materializing the whole score matrix,
/// as the baseline `windowed_attention` (with `window == grid size`) is
/// benchmarked against.
fn naive_full_attention(x: &Array2<f32>) -> Array2<f32> {
    let n = x.nrows();
    let d = x.ncols();
    let scale = 1.0 / (d as f32).sqrt();
    let scores = x.dot(&x.t()).mapv(|s| s * scale);
    let mut out = Array2::<f32>::zeros((n, d));
    for i in 0..n {
        let row = scores.row(i);
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = row.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        for (t, &e) in exp.iter().enumerate() {
            let p = e / sum;
            for j in 0..d {
                out[[i, j]] += p * x[[t, j]];
            }
        }
    }
    out
}

fn bench_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention");
    // Grid sizes representative of ViT-Base-ish patch counts (14x14=196,
    // 28x28=784) at a small window (7x7=49 tokens/window, as in Swin-T).
    for &(h, w) in &[(14usize, 14usize), (28, 28)] {
        let d = 96;
        let x = Array2::from_shape_fn((h * w, d), |(i, j)| ((i * 13 + j * 7) % 97) as f32 * 0.01);

        group.bench_with_input(BenchmarkId::new("naive_full", format!("{h}x{w}")), &x, |b, x| {
            b.iter(|| naive_full_attention(black_box(x)))
        });
        group.bench_with_input(
            BenchmarkId::new("windowed_flash_w7", format!("{h}x{w}")),
            &x,
            |b, x| b.iter(|| windowed_attention(black_box(x), h, w, 7, 32)),
        );
    }
    group.finish();
}

fn bench_token_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_merge");
    for &n in &[196usize, 784] {
        let d = 384;
        let tokens = Array2::from_shape_fn((n, d), |(i, j)| ((i * 5 + j) % 53) as f32 * 0.02);
        let r = n / 4; // merge away 25% of tokens, ToMe's typical setting
        group.bench_with_input(BenchmarkId::new("merge_r_quarter", n), &tokens, |b, t| {
            b.iter(|| token_merge(black_box(t), r))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_attention, bench_token_merge);
criterion_main!(benches);
