use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use reqwest::{Client, StatusCode, header::RANGE};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};
use tokio::{fs, io::AsyncWriteExt};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelKind {
    #[serde(rename = "1b")]
    OneB,
    #[serde(rename = "8b")]
    EightB,
}

impl ModelKind {
    pub fn slug(self) -> &'static str {
        match self {
            Self::OneB => "1b",
            Self::EightB => "8b",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::OneB => "StarVector 1B",
            Self::EightB => "StarVector 8B",
        }
    }

    fn repo(self) -> &'static str {
        match self {
            Self::OneB => "starvector/starvector-1b-im2svg",
            Self::EightB => "starvector/starvector-8b-im2svg",
        }
    }

    fn revision(self) -> &'static str {
        match self {
            Self::OneB => "380ab95d25a8e9ab1dc825debe238b4953ae13b9",
            Self::EightB => "518beea8dcb5f7a37c5911e92d1d62a76beee7f9",
        }
    }

    fn files(self) -> &'static [ModelFile] {
        match self {
            Self::OneB => FILES_1B,
            Self::EightB => FILES_8B,
        }
    }

    pub fn total_bytes(self) -> u64 {
        self.files().iter().map(|file| file.size).sum()
    }
}

#[derive(Debug, Clone, Copy)]
struct ModelFile {
    name: &'static str,
    size: u64,
}

const COMMON_1B: &[ModelFile] = &[
    ModelFile {
        name: "added_tokens.json",
        size: 99,
    },
    ModelFile {
        name: "config.json",
        size: 922,
    },
    ModelFile {
        name: "merges.txt",
        size: 441_810,
    },
    ModelFile {
        name: "model.safetensors.index.json",
        size: 65_681,
    },
    ModelFile {
        name: "preprocessor_config.json",
        size: 205,
    },
    ModelFile {
        name: "processor_config.json",
        size: 194,
    },
    ModelFile {
        name: "special_tokens_map.json",
        size: 1_012,
    },
    ModelFile {
        name: "tokenizer.json",
        size: 3_475_977,
    },
    ModelFile {
        name: "tokenizer_config.json",
        size: 4_941,
    },
    ModelFile {
        name: "vocab.json",
        size: 776_993,
    },
];

const FILES_1B: &[ModelFile] = &[
    COMMON_1B[0],
    COMMON_1B[1],
    COMMON_1B[2],
    ModelFile {
        name: "model-00001-of-00002.safetensors",
        size: 4_995_740_600,
    },
    ModelFile {
        name: "model-00002-of-00002.safetensors",
        size: 146_964_720,
    },
    COMMON_1B[3],
    COMMON_1B[4],
    COMMON_1B[5],
    COMMON_1B[6],
    COMMON_1B[7],
    COMMON_1B[8],
    COMMON_1B[9],
];

const FILES_8B: &[ModelFile] = &[
    ModelFile {
        name: "added_tokens.json",
        size: 121,
    },
    ModelFile {
        name: "config.json",
        size: 715,
    },
    ModelFile {
        name: "merges.txt",
        size: 441_705,
    },
    ModelFile {
        name: "model-00001-of-00004.safetensors",
        size: 4_889_586_776,
    },
    ModelFile {
        name: "model-00002-of-00004.safetensors",
        size: 4_946_285_040,
    },
    ModelFile {
        name: "model-00003-of-00004.safetensors",
        size: 4_999_851_312,
    },
    ModelFile {
        name: "model-00004-of-00004.safetensors",
        size: 178_570_912,
    },
    ModelFile {
        name: "model.safetensors.index.json",
        size: 105_545,
    },
    ModelFile {
        name: "preprocessor_config.json",
        size: 394,
    },
    ModelFile {
        name: "processor_config.json",
        size: 65,
    },
    ModelFile {
        name: "special_tokens_map.json",
        size: 1_438,
    },
    ModelFile {
        name: "tokenizer_config.json",
        size: 8_833,
    },
    ModelFile {
        name: "vocab.json",
        size: 973_812,
    },
];

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DownloadPhase {
    Missing,
    Downloading,
    Ready,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: &'static str,
    pub label: &'static str,
    pub selected: bool,
    pub installed: bool,
    pub phase: DownloadPhase,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelCatalog {
    pub model_admin_enabled: bool,
    pub models: Vec<ModelInfo>,
    pub runtime_device: &'static str,
    pub hardware_note: &'static str,
}

#[derive(Debug, Clone)]
struct Progress {
    phase: DownloadPhase,
    downloaded_bytes: u64,
    message: Option<String>,
}

pub struct ModelManager {
    root: PathBuf,
    selected: RwLock<ModelKind>,
    progress: RwLock<HashMap<ModelKind, Progress>>,
    client: Client,
}

impl ModelManager {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let configured = std::env::var("VECTOR_MODEL")
            .ok()
            .or_else(|| std::fs::read_to_string(root.join(".selected-model")).ok());
        let selected = configured
            .map(|value| value.trim() == "1b")
            .map(|one_b| {
                if one_b {
                    ModelKind::OneB
                } else {
                    ModelKind::EightB
                }
            })
            .unwrap_or_else(|| {
                if is_installed_sync(&root, ModelKind::OneB) {
                    ModelKind::OneB
                } else {
                    ModelKind::EightB
                }
            });
        Self {
            root,
            selected: RwLock::new(selected),
            progress: RwLock::new(HashMap::new()),
            client: Client::new(),
        }
    }

    pub fn selected(&self) -> ModelKind {
        *self.selected.read().expect("model selection lock")
    }

    pub fn select(&self, kind: ModelKind) {
        *self.selected.write().expect("model selection lock") = kind;
        let _ = std::fs::create_dir_all(&self.root);
        let _ = std::fs::write(self.root.join(".selected-model"), kind.slug());
    }

    pub fn model_dir(&self, kind: ModelKind) -> PathBuf {
        self.root.join(format!("starvector-{}-im2svg", kind.slug()))
    }

    pub async fn is_installed(&self, kind: ModelKind) -> bool {
        for file in kind.files() {
            let Ok(metadata) = fs::metadata(self.model_dir(kind).join(file.name)).await else {
                return false;
            };
            if metadata.len() != file.size {
                return false;
            }
        }
        true
    }

    pub async fn delete(&self, kind: ModelKind) -> Result<()> {
        if matches!(
            self.progress
                .read()
                .expect("download progress lock")
                .get(&kind)
                .map(|progress| &progress.phase),
            Some(DownloadPhase::Downloading)
        ) {
            bail!("{} is still downloading", kind.label());
        }
        match fs::remove_dir_all(self.model_dir(kind)).await {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("delete local model checkpoint"),
        }
        self.progress
            .write()
            .expect("download progress lock")
            .remove(&kind);
        if self.selected() == kind {
            self.select(match kind {
                ModelKind::OneB => ModelKind::EightB,
                ModelKind::EightB => ModelKind::OneB,
            });
        }
        Ok(())
    }

    pub async fn catalog(&self) -> ModelCatalog {
        let selected = self.selected();
        let snapshot = self
            .progress
            .read()
            .expect("download progress lock")
            .clone();
        let mut models = Vec::new();
        for kind in [ModelKind::OneB, ModelKind::EightB] {
            let installed = self.is_installed(kind).await;
            let progress = snapshot.get(&kind);
            models.push(ModelInfo {
                id: kind.slug(),
                label: kind.label(),
                selected: kind == selected,
                installed,
                phase: if installed {
                    DownloadPhase::Ready
                } else {
                    progress
                        .map(|p| p.phase.clone())
                        .unwrap_or(DownloadPhase::Missing)
                },
                downloaded_bytes: progress.map(|p| p.downloaded_bytes).unwrap_or(0),
                total_bytes: kind.total_bytes(),
                message: progress.and_then(|p| p.message.clone()),
            });
        }
        ModelCatalog {
            model_admin_enabled: std::env::var_os("VECTOR_ENABLE_MODEL_ADMIN").is_some(),
            models,
            runtime_device: runtime_device_label(),
            hardware_note: if cfg!(feature = "cuda") {
                "NVIDIA CUDA detected. The hosted service defaults to StarVector 8B on a high-memory GPU."
            } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                "Apple Silicon detected. Metal/BF16 is preferred with CPU fallback."
            } else {
                "This host is not Apple Silicon; StarVector uses CPU and 8B may exceed practical memory or time limits."
            },
        }
    }

    pub fn start_download(self: &Arc<Self>, kind: ModelKind) -> Result<()> {
        {
            let mut progress = self.progress.write().expect("download progress lock");
            if matches!(
                progress.get(&kind).map(|p| &p.phase),
                Some(DownloadPhase::Downloading)
            ) {
                bail!("{} download is already running", kind.label());
            }
            progress.insert(
                kind,
                Progress {
                    phase: DownloadPhase::Downloading,
                    downloaded_bytes: 0,
                    message: Some("Starting download…".to_owned()),
                },
            );
        }
        let manager = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(error) = manager.download(kind).await {
                manager
                    .progress
                    .write()
                    .expect("download progress lock")
                    .insert(
                        kind,
                        Progress {
                            phase: DownloadPhase::Failed,
                            downloaded_bytes: 0,
                            message: Some(error.to_string()),
                        },
                    );
            }
        });
        Ok(())
    }

    /// Download every missing checkpoint before accepting public traffic.
    /// Existing complete files are retained, so this is safe to run on every
    /// container start when `VECTOR_MODEL_DIR` points to persistent storage.
    pub async fn bootstrap_all(&self) -> Result<()> {
        for kind in [ModelKind::OneB, ModelKind::EightB] {
            if !self.is_installed(kind).await {
                self.download(kind).await?;
            }
        }
        Ok(())
    }

    async fn download(&self, kind: ModelKind) -> Result<()> {
        let model_dir = self.model_dir(kind);
        fs::create_dir_all(&model_dir)
            .await
            .context("create model directory")?;
        let mut completed = 0u64;
        for file in kind.files() {
            let destination = model_dir.join(file.name);
            if fs::metadata(&destination)
                .await
                .map(|m| m.len())
                .unwrap_or(0)
                == file.size
            {
                completed += file.size;
                self.set_progress(kind, completed, Some(file.name));
                continue;
            }
            self.download_file(kind, file, &destination, completed)
                .await?;
            completed += file.size;
            self.set_progress(kind, completed, Some(file.name));
        }
        self.progress
            .write()
            .expect("download progress lock")
            .insert(
                kind,
                Progress {
                    phase: DownloadPhase::Ready,
                    downloaded_bytes: kind.total_bytes(),
                    message: Some("Checkpoint ready".to_owned()),
                },
            );
        Ok(())
    }

    async fn download_file(
        &self,
        kind: ModelKind,
        file: &ModelFile,
        destination: &Path,
        base: u64,
    ) -> Result<()> {
        let part = destination.with_extension("part");
        let existing = fs::metadata(&part)
            .await
            .map(|m| m.len())
            .unwrap_or(0)
            .min(file.size);
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            kind.repo(),
            kind.revision(),
            file.name
        );
        let mut request = self.client.get(url);
        if existing > 0 {
            request = request.header(RANGE, format!("bytes={existing}-"));
        }
        let response = request.send().await.context("connect to Hugging Face")?;
        let append = response.status() == StatusCode::PARTIAL_CONTENT && existing > 0;
        if !response.status().is_success() {
            bail!(
                "Hugging Face returned {} for {}",
                response.status(),
                file.name
            );
        }
        let mut output = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(&part)
            .await
            .context("open partial model file")?;
        let mut received = if append { existing } else { 0 };
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("read model download")?;
            output.write_all(&chunk).await.context("write model data")?;
            received += chunk.len() as u64;
            self.set_progress(kind, base + received, Some(file.name));
        }
        output.flush().await.context("flush model file")?;
        if received != file.size {
            bail!(
                "{} is incomplete: expected {} bytes, received {}",
                file.name,
                file.size,
                received
            );
        }
        fs::rename(part, destination)
            .await
            .context("finish model file")?;
        Ok(())
    }

    fn set_progress(&self, kind: ModelKind, bytes: u64, file: Option<&str>) {
        self.progress
            .write()
            .expect("download progress lock")
            .insert(
                kind,
                Progress {
                    phase: DownloadPhase::Downloading,
                    downloaded_bytes: bytes,
                    message: file.map(|name| format!("Downloading {name}")),
                },
            );
    }
}

fn is_installed_sync(root: &Path, kind: ModelKind) -> bool {
    kind.files().iter().all(|file| {
        std::fs::metadata(
            root.join(format!("starvector-{}-im2svg", kind.slug()))
                .join(file.name),
        )
        .map(|metadata| metadata.len() == file.size)
        .unwrap_or(false)
    })
}

pub fn runtime_device_label() -> &'static str {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "Metal · BF16"
    } else if cfg!(feature = "cuda") {
        "NVIDIA CUDA · GPU"
    } else {
        "CPU · F32"
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelKind, ModelManager};

    #[test]
    fn model_ids_match_the_api_contract() {
        assert_eq!(
            serde_json::from_str::<ModelKind>("\"1b\"").unwrap(),
            ModelKind::OneB
        );
        assert_eq!(
            serde_json::from_str::<ModelKind>("\"8b\"").unwrap(),
            ModelKind::EightB
        );
    }

    #[tokio::test]
    async fn deleting_a_model_removes_only_its_checkpoint_directory() {
        let root = tempfile::tempdir().unwrap();
        let manager = ModelManager::new(root.path());
        manager.select(ModelKind::OneB);
        let model_dir = manager.model_dir(ModelKind::OneB);
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("checkpoint-marker"), b"local model").unwrap();
        std::fs::write(root.path().join("keep-me"), b"unrelated").unwrap();

        manager.delete(ModelKind::OneB).await.unwrap();

        assert!(!model_dir.exists());
        assert!(root.path().join("keep-me").exists());
        assert_eq!(manager.selected(), ModelKind::EightB);
    }
}
