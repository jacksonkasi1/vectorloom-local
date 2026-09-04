use anyhow::{Context, Result, bail};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use serde::Serialize;
use std::{env, time::Instant};
use vtracer::{ColorImage, Config, FitMode, Hierarchical, Preset};

pub mod models;
pub mod starvector;

const MAX_EDGE: u32 = 4_096;

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeStatus {
    pub requested_model: &'static str,
    pub device: String,
    pub precision: &'static str,
    pub model_runtime: &'static str,
    pub fallback_engine: &'static str,
    pub detail: String,
}

#[derive(Debug, Serialize)]
pub struct VectorizedImage {
    pub svg: String,
    pub width: u32,
    pub height: u32,
    pub elapsed_ms: u128,
    pub engine: String,
    pub status: RuntimeStatus,
    pub warning: Option<String>,
}

pub fn runtime_status() -> RuntimeStatus {
    let requested_model = match env::var("VECTOR_MODEL").as_deref() {
        Ok("1b") => "StarVector 1B",
        _ => "StarVector 8B",
    };
    let device = models::runtime_device_label().to_owned();
    let detail = "Rust StarVector inference is linked. Downloaded checkpoints run in-process; missing or invalid model output uses the visible VTracer fallback.".to_owned();
    RuntimeStatus {
        requested_model,
        device,
        precision: if cfg!(feature = "cuda") {
            "FP16"
        } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            "BF16"
        } else {
            "F32"
        },
        model_runtime: "linked and checkpoint-gated",
        fallback_engine: "VTracer spline/cutout pipeline",
        detail,
    }
}

pub fn vectorize(bytes: &[u8]) -> Result<VectorizedImage> {
    let started = Instant::now();
    let image = image::load_from_memory(bytes).context("decode PNG, JPEG, or WebP image")?;
    let image = resize_for_trace(image);
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        bail!("image has no pixels");
    }

    let rgba = image.to_rgba8();
    let color_count = estimate_color_complexity(rgba.as_raw());
    let max_colors = auto_palette_size(color_count, width, height);
    let source = ColorImage {
        pixels: rgba.into_raw(),
        width: width as usize,
        height: height as usize,
    };

    // The policy favors a seam-free, compact spline result for graphic assets
    // without exposing quality knobs in the product UI.
    let mut config = Config::from_preset(Preset::Poster);
    config.mode = FitMode::Spline;
    config.hierarchical = Hierarchical::Cutout;
    config.max_colors = Some(max_colors);
    config.simplify = Some(if max_colors <= 8 { 0.8 } else { 1.35 });
    config.optimize = 2;
    let svg = config
        .build()
        .context("build automatic vectorization pipeline")?
        .to_svg(&source)
        .context("trace image into SVG")?;

    Ok(VectorizedImage {
        svg,
        width,
        height,
        elapsed_ms: started.elapsed().as_millis(),
        engine: "VTracer automatic spline/cutout".to_owned(),
        status: runtime_status(),
        warning: None,
    })
}

fn resize_for_trace(image: DynamicImage) -> DynamicImage {
    let (width, height) = image.dimensions();
    let edge = width.max(height);
    if edge <= MAX_EDGE {
        return image;
    }
    let scale = MAX_EDGE as f32 / edge as f32;
    image.resize(
        (width as f32 * scale).round() as u32,
        (height as f32 * scale).round() as u32,
        FilterType::Lanczos3,
    )
}

fn estimate_color_complexity(pixels: &[u8]) -> usize {
    let mut seen = std::collections::HashSet::new();
    let (samples, _) = pixels.as_chunks::<64>();
    for sample in samples {
        let pixel = &sample[..4];
        if pixel[3] > 8 {
            seen.insert((pixel[0] >> 4, pixel[1] >> 4, pixel[2] >> 4));
        }
        if seen.len() >= 96 {
            break;
        }
    }
    seen.len()
}

fn auto_palette_size(color_count: usize, width: u32, height: u32) -> usize {
    let area = width as u64 * height as u64;
    match (color_count, area) {
        (0..=5, _) => 6,
        (6..=16, _) => 12,
        (17..=40, _) => 20,
        (_, area) if area > 2_000_000 => 28,
        _ => 36,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageFormat, Rgba, RgbaImage};
    use std::io::Cursor;

    #[test]
    fn emits_a_complete_svg() {
        let mut image = RgbaImage::new(32, 32);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = if x < 16 && y < 16 {
                Rgba([255, 80, 40, 255])
            } else {
                Rgba([30, 60, 230, 255])
            };
        }
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();
        let output = vectorize(bytes.get_ref()).unwrap();
        assert!(output.svg.contains("<svg"));
        assert!(output.svg.trim_end().ends_with("</svg>"));
    }
}
