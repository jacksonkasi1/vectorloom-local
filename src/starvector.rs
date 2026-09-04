use crate::models::ModelKind;
use anyhow::{Context, Result};
use starvector_rs::{GenerationConfig, PrecisionPolicy, RuntimeDevice, StarVector};
use std::{io::Write, path::Path, process::Command, sync::Mutex, time::Instant};

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
        if matches!(kind, ModelKind::EightB)
            && std::env::var_os("VECTOR_OFFICIAL_8B_RUNTIME").is_some()
        {
            return generate_with_official_runtime(model_dir, image, started);
        }
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
            // The released 8B checkpoint needs sampling to emit meaningful
            // image-conditioned SVG; validation below repairs only its known
            // truncated/malformed-element edge cases.
            do_sample: matches!(kind, ModelKind::EightB),
            temperature: if matches!(kind, ModelKind::EightB) {
                0.7
            } else {
                0.2
            },
            top_p: 0.95,
            repetition_penalty: 1.0,
            // Favor the closing SVG token without forcing it early.
            svg_stop_bias: 5.0,
            seed: 42,
        };
        let output = loaded
            .model
            .generate_svg(temp.path(), &generation)
            .context("generate SVG")?;
        if let Ok(path) = std::env::var("VECTOR_DEBUG_RAW_OUTPUT") {
            let _ = std::fs::write(path, &output.svg);
        }
        Ok(StarVectorResult {
            // 8B reliably draws the whole image but can consume its context
            // before emitting closing XML tags. Recover only that known model
            // case; 1B remains strict because its partial output is unreliable.
            svg: validate_svg(output.svg, matches!(kind, ModelKind::EightB))?,
            elapsed_ms: started.elapsed().as_millis(),
            engine: format!("{} · {}", kind.label(), loaded.device_name),
        })
    }
}

fn generate_with_official_runtime(
    model_dir: &Path,
    image: &[u8],
    started: Instant,
) -> Result<StarVectorResult> {
    let mut input = tempfile::Builder::new().suffix(".png").tempfile()?;
    input.write_all(image)?;
    let output = tempfile::Builder::new().suffix(".svg").tempfile()?;
    let status = Command::new("python3")
        .args([
            "/app/reference_vectorize.py",
            &input.path().display().to_string(),
            &output.path().display().to_string(),
            &model_dir.display().to_string(),
        ])
        .status()
        .context("start official StarVector 8B runtime")?;
    anyhow::ensure!(status.success(), "official StarVector 8B runtime failed");
    let raw = std::fs::read_to_string(output.path()).context("read official StarVector SVG")?;
    Ok(StarVectorResult {
        svg: validate_svg(raw, true)?,
        elapsed_ms: started.elapsed().as_millis(),
        engine: "StarVector 8B · official Transformers CUDA".to_owned(),
    })
}

fn validate_svg(raw: String, allow_tag_recovery: bool) -> Result<String> {
    let start = raw.find("<svg").context("model output has no SVG root")?;
    let svg = match raw.rfind("</svg>") {
        Some(close) => raw[start..close + "</svg>".len()].trim().to_owned(),
        None if allow_tag_recovery => balance_svg_tags(&raw[start..])?,
        None => anyhow::bail!("model output is incomplete (missing </svg>)"),
    };
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
    let svg = match roxmltree::Document::parse(&svg) {
        Ok(_) => svg,
        Err(_) if allow_tag_recovery => balance_svg_tags(&remove_malformed_tags(&svg))?,
        Err(error) => return Err(error).context("model output is not valid XML"),
    };
    let document = roxmltree::Document::parse(&svg).context("model output is not valid XML")?;
    anyhow::ensure!(
        document.root_element().tag_name().name() == "svg",
        "model output root is not SVG"
    );
    Ok(svg)
}

/// Drop only an unterminated element when the decoder emits a new `<` before
/// closing a quoted attribute. Valid tags and all following elements remain.
fn remove_malformed_tags(raw: &str) -> String {
    let mut clean = String::with_capacity(raw.len());
    let mut cursor = 0;
    while let Some(relative_start) = raw[cursor..].find('<') {
        let start = cursor + relative_start;
        clean.push_str(&raw[cursor..start]);
        let mut quote = None;
        let mut end = None;
        let mut nested = None;
        for (relative, ch) in raw[start + 1..].char_indices() {
            let index = start + 1 + relative;
            match (quote, ch) {
                (Some(current), value) if value == current => quote = None,
                (None, '\"' | '\'') => quote = Some(ch),
                (Some(_), '<') => {
                    nested = Some(index);
                    break;
                }
                (None, '>') => {
                    end = Some(index);
                    break;
                }
                _ => {}
            }
        }
        if let Some(end) = end {
            clean.push_str(&raw[start..=end]);
            cursor = end + 1;
        } else if let Some(next) = nested {
            cursor = next;
        } else {
            break;
        }
    }
    clean.push_str(&raw[cursor..]);
    clean
}

fn balance_svg_tags(raw: &str) -> Result<String> {
    let end = raw
        .rfind('>')
        .context("model output has no complete SVG element")?;
    let complete = &raw[..=end];
    let mut open = Vec::new();
    let mut offset = 0;
    while let Some(relative_start) = complete[offset..].find('<') {
        let start = offset + relative_start;
        let Some(relative_end) = complete[start..].find('>') else {
            break;
        };
        let end = start + relative_end;
        let tag = complete[start + 1..end].trim();
        offset = end + 1;
        if tag.is_empty() || tag.starts_with('!') || tag.starts_with('?') || tag.ends_with('/') {
            continue;
        }
        if let Some(name) = tag.strip_prefix('/') {
            if open
                .last()
                .is_some_and(|value| value == name.split_whitespace().next().unwrap_or_default())
            {
                open.pop();
            }
        } else if let Some(name) = tag
            .split_whitespace()
            .next()
            .filter(|name| !name.is_empty())
        {
            open.push(name.to_owned());
        }
    }
    anyhow::ensure!(
        open.first().is_some_and(|tag| tag == "svg"),
        "model output has no SVG root"
    );
    let mut recovered = complete.to_owned();
    while let Some(tag) = open.pop() {
        recovered.push_str(&format!("</{tag}>"));
    }
    Ok(recovered)
}

#[cfg(test)]
mod tests {
    use super::validate_svg;

    #[test]
    fn accepts_complete_svg_and_removes_outer_text() {
        let svg = validate_svg(
            "noise <svg xmlns=\"http://www.w3.org/2000/svg\"><path d=\"M0 0\"/></svg> tail"
                .to_owned(),
            false,
        )
        .unwrap();
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn rejects_incomplete_or_active_content() {
        assert!(validate_svg("not an svg".to_owned(), false).is_err());
        assert!(validate_svg("<svg><script>alert(1)</script></svg>".to_owned(), false).is_err());
    }

    #[test]
    fn rejects_incomplete_model_svg() {
        assert!(validate_svg("<svg><path d=\"M0 0\"/>".to_owned(), false).is_err());
    }
}
