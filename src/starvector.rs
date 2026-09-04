use crate::models::ModelKind;
use anyhow::{Context, Result};
use starvector_rs::{GenerationConfig, PrecisionPolicy, RuntimeDevice, StarVector};
use std::{io::Write, path::Path, sync::Mutex, time::Instant};

pub struct StarVectorResult {
    pub svg: String,
    pub elapsed_ms: u128,
    pub engine: String,
}

struct LoadedModel {
    kind: ModelKind,
    model: StarVector,
    device_name: &'static str,
}

pub struct StarVectorRuntime {
    loaded: Mutex<Option<LoadedModel>>,
}

impl Default for StarVectorRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl StarVectorRuntime {
    pub fn new() -> Self {
        Self {
            loaded: Mutex::new(None),
        }
    }

    pub fn unload(&self, kind: ModelKind) {
        let mut loaded = self.loaded.lock().expect("StarVector runtime lock");
        if loaded.as_ref().map(|value| value.kind) == Some(kind) {
            *loaded = None;
        }
    }

    pub fn generate(
        &self,
        kind: ModelKind,
        model_dir: &Path,
        image: &[u8],
    ) -> Result<StarVectorResult> {
        let started = Instant::now();
        let mut loaded = self.loaded.lock().expect("StarVector runtime lock");
        if loaded.as_ref().map(|value| value.kind) != Some(kind) {
            let runtime_device = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                RuntimeDevice::Metal(0)
            } else if cfg!(feature = "cuda") {
                RuntimeDevice::Cuda(0)
            } else {
                RuntimeDevice::Cpu
            };
            let device_name = match runtime_device {
                RuntimeDevice::Metal(_) => "Metal/BF16",
                RuntimeDevice::Cuda(_) => "NVIDIA CUDA",
                RuntimeDevice::Cpu => "CPU/F32",
            };
            let device = runtime_device
                .to_candle_device()
                .context("initialize inference device")?;
            let precision = PrecisionPolicy::for_device(&device);
            let model = StarVector::load(model_dir, &device, precision)
                .context("load StarVector checkpoint")?;
            *loaded = Some(LoadedModel {
                kind,
                model,
                device_name,
            });
        }

        let loaded = loaded.as_mut().expect("loaded StarVector model");
        let mut temp = tempfile::Builder::new()
            .suffix(".png")
            .tempfile()
            .context("create local image buffer")?;
        temp.write_all(image).context("stage image for inference")?;
        let max_new_tokens = std::env::var("VECTOR_MAX_TOKENS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(7_933);
        let generation = GenerationConfig {
            max_new_tokens,
            num_beams: 1,
            do_sample: matches!(kind, ModelKind::EightB),
            temperature: if matches!(kind, ModelKind::EightB) {
                0.7
            } else {
                0.2
            },
            top_p: 0.95,
            repetition_penalty: 1.0,
            // Both released im2svg checkpoints can otherwise exhaust their
            // output budget without emitting the closing `</svg>` token.
            // Favoring that token preserves a complete, valid document while
            // still letting the model decide when the drawing is finished.
            svg_stop_bias: 5.0,
            seed: 42,
        };
        let output = loaded
            .model
            .generate_svg(temp.path(), &generation)
            .context("generate SVG")?;
        Ok(StarVectorResult {
            svg: validate_svg(output.svg)?,
            elapsed_ms: started.elapsed().as_millis(),
            engine: format!("{} · {}", kind.label(), loaded.device_name),
        })
    }
}

fn validate_svg(raw: String) -> Result<String> {
    let start = raw.find("<svg").context("model output has no SVG root")?;
    let close = raw
        .rfind("</svg>")
        .context("model output is incomplete (missing </svg>)")?;
    let svg = raw[start..close + "</svg>".len()].trim().to_owned();
    let lower = svg.to_ascii_lowercase();
    for forbidden in [
        "<script",
        "<foreignobject",
        "<image",
        "javascript:",
        "onload=",
    ] {
        anyhow::ensure!(
            !lower.contains(forbidden),
            "model output contains forbidden SVG content"
        );
    }
    let document = roxmltree::Document::parse(&svg).context("model output is not valid XML")?;
    anyhow::ensure!(
        document.root_element().tag_name().name() == "svg",
        "model output root is not SVG"
    );
    Ok(svg)
}

#[cfg(test)]
mod tests {
    use super::validate_svg;

    #[test]
    fn accepts_complete_svg_and_removes_outer_text() {
        let svg = validate_svg(
            "noise <svg xmlns=\"http://www.w3.org/2000/svg\"><path d=\"M0 0\"/></svg> tail"
                .to_owned(),
        )
        .unwrap();
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn rejects_incomplete_or_active_content() {
        assert!(validate_svg("not an svg".to_owned()).is_err());
        assert!(validate_svg("<svg><script>alert(1)</script></svg>".to_owned()).is_err());
    }

    #[test]
    fn rejects_incomplete_model_svg() {
        assert!(validate_svg("<svg><path d=\"M0 0\"/>".to_owned()).is_err());
    }
}
