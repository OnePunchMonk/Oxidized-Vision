//! Fused image preprocessing: decode -> SIMD resize -> normalize -> NCHW.
//!
//! In real vision-serving deployments, decode/resize/normalize of the raw
//! input image is often the actual latency bottleneck, not the model
//! forward pass itself — a naive per-pixel resize loop in scalar Rust (or,
//! worse, round-tripping the image through Python for preprocessing) can
//! dominate end-to-end request latency for small/fast vision models.
//!
//! This module decodes with the `image` crate and resizes with
//! `fast_image_resize`, which uses SIMD (SSE4.1/AVX2 on x86, NEON on ARM)
//! convolution-based resize kernels instead of scalar bilinear/nearest
//! loops, then fuses the u8 -> normalized-f32 NCHW conversion into a single
//! pass over the resized buffer.

use anyhow::{Context, Result};
use fast_image_resize as fr;
use image::GenericImageView;
use ndarray::{ArrayD, IxDyn};

/// Per-channel normalization stats. Defaults match common ImageNet-style
/// preprocessing (mean/std for RGB in [0, 1] pixel range).
#[derive(Debug, Clone, Copy)]
pub struct NormalizeStats {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for NormalizeStats {
    fn default() -> Self {
        Self {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}

/// Decode raw image bytes (JPEG/PNG/etc, whatever `image` supports), resize
/// to `(target_h, target_w)` with a SIMD-accelerated Lanczos3 filter, and
/// normalize into a `[1, 3, target_h, target_w]` NCHW `f32` tensor in one
/// fused pass — no intermediate `Vec<f32>` allocation beyond the final
/// output buffer.
pub fn decode_resize_normalize(
    bytes: &[u8],
    target_h: u32,
    target_w: u32,
    stats: NormalizeStats,
) -> Result<ArrayD<f32>> {
    let decoded = image::load_from_memory(bytes).context("Failed to decode image")?;
    let (src_w, src_h) = decoded.dimensions();

    let src_image = fr::images::Image::from_vec_u8(
        src_w,
        src_h,
        decoded.to_rgb8().into_raw(),
        fr::PixelType::U8x3,
    )
    .context("Failed to wrap decoded image for resizing")?;

    let mut dst_image = fr::images::Image::new(target_w, target_h, fr::PixelType::U8x3);

    let mut resizer = fr::Resizer::new();
    let resize_options =
        fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Convolution(fr::FilterType::Lanczos3));
    resizer
        .resize(&src_image, &mut dst_image, &resize_options)
        .context("SIMD resize failed")?;

    // Fused u8 RGB HWC -> normalized f32 CHW, single pass over the resized buffer.
    let hw = (target_h * target_w) as usize;
    let mut chw = vec![0f32; 3 * hw];
    let (chunks, _) = dst_image.buffer().as_chunks::<3>();
    for (i, px) in chunks.iter().enumerate() {
        chw[i] = (px[0] as f32 / 255.0 - stats.mean[0]) / stats.std[0];
        chw[hw + i] = (px[1] as f32 / 255.0 - stats.mean[1]) / stats.std[1];
        chw[2 * hw + i] = (px[2] as f32 / 255.0 - stats.mean[2]) / stats.std[2];
    }

    Ok(ArrayD::from_shape_vec(
        IxDyn(&[1, 3, target_h as usize, target_w as usize]),
        chw,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_resize_normalize_shape() {
        let img = image::RgbImage::from_pixel(64, 48, image::Rgb([128, 64, 200]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();

        let out = decode_resize_normalize(&bytes, 32, 32, NormalizeStats::default()).unwrap();
        assert_eq!(out.shape(), &[1, 3, 32, 32]);
    }

    #[test]
    fn test_decode_invalid_bytes_errors() {
        let result = decode_resize_normalize(b"not an image", 32, 32, NormalizeStats::default());
        assert!(result.is_err());
    }
}
