# VectorLoom

Image vectorization for Apple Silicon and hosted NVIDIA GPUs. The UI is deliberately simple: upload a PNG, JPG, or WebP; choose StarVector 1B or 8B; preview and download the SVG.

## Hosted GPU deployment

The included Docker image compiles Candle with CUDA and serves both the browser UI and HTTP API. It automatically installs both model checkpoints into `VECTOR_MODEL_DIR` (default `/models`) before accepting traffic. Mount persistent storage there: the two models need about 20 GB together.

```sh
docker build -t vectorloom:cuda .
docker run --gpus all -p 3000:3000 -v vectorloom-models:/models vectorloom:cuda
```

See [deploy/README.md](deploy/README.md) for Modal deployment. The image is designed for an A100 80 GB so either model can be selected safely.

## Install the macOS app

Download `VectorLoom-macOS-universal.dmg`, open it, and drag `VectorLoom.app` to Applications. A ZIP is also provided. The app is Universal: Apple Silicon runs native Metal/BF16 inference, while Intel Macs use Accelerate CPU/F32. Because the development build is ad-hoc signed rather than Apple-notarized, the first launch may require right-clicking the app and choosing **Open**.

The app downloads models on demand and stores them in `~/Library/Application Support/VectorLoom/models`. Downloaded checkpoints survive app upgrades. Use **Delete model** to remove a local checkpoint and reclaim its 4.8 GiB or 14.0 GiB; VectorLoom asks for confirmation first.

Build the Universal app and installer locally with:

```sh
./scripts/build-macos-app.sh
```

Outputs are written to `dist/VectorLoom.app`, `dist/VectorLoom-macOS-universal.dmg`, and `dist/VectorLoom-macOS-universal.zip`.

## Run from source

```sh
cargo run --release
open http://127.0.0.1:3000
```

On macOS, double-click `VectorLoom.command` for the same release-mode startup.

The model panel downloads official, revision-pinned Hugging Face checkpoints directly to `models/` with resumable partial files:

- StarVector 1B: 4.8 GiB, the practical debugging/faster option.
- StarVector 8B: 14.0 GiB, the default quality option for the intended 64 GB Apple Silicon machine.

Selecting a model is remembered across restarts. When there is no saved choice, an already-downloaded 1B checkpoint is preferred over an unavailable 8B checkpoint. Selecting a downloaded model makes uploads use real in-process Candle inference; the loaded model is cached between requests.

Images and generated SVGs are processed in memory and are not persisted. Local builds bind to `127.0.0.1` by default; hosted containers set `VECTOR_BIND=0.0.0.0`.

## Current inference policy

- `VECTOR_MODEL=8b` and `VECTOR_MODEL=1b` can override the saved model choice at startup.
- The included runtime uses the Metal-enabled `jacksonkasi1/starvector-rs` fork for real in-process 1B/8B inference and validates generated XML before download.
- Apple Silicon selects Candle Metal/BF16 automatically. Intel Macs use CPU/F32 with Apple's Accelerate framework.
- If a checkpoint is missing or inference fails, VTracer's Rust spline/cutout pipeline remains available and the UI shows the fallback reason.

StarVector generates SVG text one token at a time. The first conversion also loads several gigabytes of weights; later conversions reuse the loaded model and are substantially faster. The UI reports the full wait as a live `h m s` timer. For development-only bounded runs, `VECTOR_MAX_TOKENS=512 cargo run --release` limits generation, but too small a value can truncate complex SVGs and trigger the visible tracer fallback.

See [STARVECTOR_RESEARCH.md](STARVECTOR_RESEARCH.md) for the investigation and implementation boundary.

## Validate

```sh
cargo test
cargo clippy --all-targets -- -D warnings
```

## License

Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream notices.
