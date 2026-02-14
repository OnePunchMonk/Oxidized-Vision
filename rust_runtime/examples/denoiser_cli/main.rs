use clap::Parser;
use runner_core::{Runner, RunnerConfig};
use runner_tract::TractRunner;
use ndarray::{ArrayD, Array3, IxDyn};
use image::{io::Reader as ImageReader, DynamicImage, Rgb32FImage};
use anyhow::Result;

#[derive(Parser, Debug)]
#[clap(author, version, about = "OxidizedVision image denoiser CLI")]
struct Args {
    /// Path to the ONNX model file
    #[clap(short, long)]
    model: String,

    /// Path to the input image
    #[clap(short, long)]
    input: String,

    /// Path to save the output image
    #[clap(short, long)]
    output: String,

    /// Target height for model input (default: use original)
    #[clap(long, default_value_t = 256)]
    height: u32,

    /// Target width for model input (default: use original)
    #[clap(long, default_value_t = 256)]
    width: u32,

    /// Normalization mean (comma-separated R,G,B)
    #[clap(long, default_value = "0.485,0.456,0.406")]
    mean: String,

    /// Normalization std (comma-separated R,G,B)
    #[clap(long, default_value = "0.229,0.224,0.225")]
    std_dev: String,
}

/// Preprocess an image: resize, convert to f32, normalize, and reshape to CHW.
fn preprocess_image(
    img: &DynamicImage,
    height: u32,
    width: u32,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Result<ArrayD<f32>> {
    // Resize image
    let resized = img.resize_exact(width, height, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb32f();

    let h = height as usize;
    let w = width as usize;

    // Convert HWC -> CHW and normalize
    let mut chw_data = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = (pixel[c] - mean[c]) / std[c];
                chw_data[c * h * w + y * w + x] = val;
            }
        }
    }

    Ok(ArrayD::from_shape_vec(IxDyn(&[1, 3, h, w]), chw_data)?)
}

/// Postprocess model output: denormalize, clamp, convert CHW -> HWC, save as image.
fn postprocess_and_save(
    output: &ArrayD<f32>,
    output_path: &str,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Result<()> {
    let shape = output.shape();
    if shape.len() != 4 {
        anyhow::bail!("Expected 4D output, got {}D", shape.len());
    }

    let h = shape[2];
    let w = shape[3];

    // Convert CHW -> HWC and denormalize
    let mut img_data = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let val = output[[0, c, y, x]];
                let denorm = val * std[c] + mean[c];
                let clamped = denorm.clamp(0.0, 1.0);
                img_data[(y * w + x) * 3 + c] = clamped;
            }
        }
    }

    // Create and save image
    let img = Rgb32FImage::from_vec(w as u32, h as u32, img_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create output image"))?;

    let dynamic = DynamicImage::ImageRgb32F(img);
    let rgb8 = dynamic.to_rgb8();
    rgb8.save(output_path)?;

    println!("✅ Output image saved to {}", output_path);
    Ok(())
}

fn parse_float3(s: &str) -> Result<[f32; 3]> {
    let parts: Vec<f32> = s.split(',').map(|v| v.trim().parse::<f32>()).collect::<std::result::Result<Vec<_>, _>>()?;
    if parts.len() != 3 {
        anyhow::bail!("Expected 3 comma-separated values, got {}", parts.len());
    }
    Ok([parts[0], parts[1], parts[2]])
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mean = parse_float3(&args.mean)?;
    let std_dev = parse_float3(&args.std_dev)?;

    println!("📦 Loading model: {}", args.model);
    let config = RunnerConfig {
        model_path: args.model.clone(),
        input_shape: vec![1, 3, args.height as usize, args.width as usize],
        use_cuda: false,
        optimize: true,
    };
    let runner = TractRunner::from_config(&config)?;

    println!("🖼️  Reading input image: {}", args.input);
    let img = ImageReader::open(&args.input)?.decode()?;
    let original_size = (img.width(), img.height());
    println!("   Original size: {}x{}", original_size.0, original_size.1);

    println!("🔄 Preprocessing (resize to {}x{}, normalize)...", args.width, args.height);
    let input = preprocess_image(&img, args.height, args.width, &mean, &std_dev)?;
    println!("   Input tensor shape: {:?}", input.shape());

    println!("🧠 Running inference...");
    let output = runner.run(&input)?;
    println!("   Output tensor shape: {:?}", output.shape());

    println!("💾 Postprocessing and saving...");
    postprocess_and_save(&output, &args.output, &mean, &std_dev)?;

    Ok(())
}
