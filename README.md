# VectorLoom

Image vectorization for Apple Silicon and hosted NVIDIA GPUs. Upload a PNG, JPG, or WebP; choose direct tracing (the default for logos), StarVector 1B, or StarVector 8B; preview and download the SVG.

## Hosted GPU deployment

The included Docker image serves both the browser UI and HTTP API. Its startup script downloads missing model checkpoints into `VECTOR_MODEL_DIR` (default `/models`) in the background. Mount persistent storage there: the two models need about 20 GB together. Direct tracing does not require either checkpoint.

```sh
docker build -t vectorloom:cuda .
docker run --gpus all -p 3000:3000 -v vectorloom-models:/models vectorloom:cuda /app/docker-entrypoint.sh
```

See [deploy/README.md](deploy/README.md) for Modal deployment. Modal prefers an L40S (48 GB), with larger compatible GPU pools as capacity fallbacks.

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

- StarVector 1B: 4.8 GiB checkpoint.
- StarVector 8B: 14.0 GiB checkpoint.

The browser starts with direct tracing selected. Selecting an AI model applies to that browser's subsequent uploads, without changing other users' choices. The API also accepts `model=trace`, `model=1b`, or `model=8b` with each upload. If the API caller omits the model, the server uses its configured AI default.

Images and generated SVGs are processed in memory and are not persisted. Local builds bind to `127.0.0.1` by default; hosted containers set `VECTOR_BIND=0.0.0.0`.

## Current inference policy

- `VECTOR_MODEL=8b` and `VECTOR_MODEL=1b` can override the saved model choice at startup.
- The included runtime uses the Metal-enabled `jacksonkasi1/starvector-rs` fork for real in-process 1B/8B inference and validates generated XML before download.
- Hosted CUDA containers use the official Transformers implementation in a persistent worker (`VECTOR_OFFICIAL_RUNTIME=1`). The 1B decoder is constructed from the installed public checkpoint and its shared output weights are re-tied after loading.
- Apple Silicon selects Candle Metal/BF16 automatically. Intel Macs use CPU/F32 with Apple's Accelerate framework.
- If a checkpoint is missing or inference fails, VTracer's Rust spline/cutout pipeline remains available and the UI shows the fallback reason.

StarVector generates SVG text one token at a time. The first AI conversion loads several gigabytes of weights; consecutive conversions with the same model reuse those weights while the server stays warm. Switching models or a server cold start requires loading again. The UI submits asynchronous jobs and polls their status, displaying the full wait as a live `h m s` timer. `VECTOR_MAX_TOKENS` limits the native Rust runtime only.

Direct tracing skips model inference and preserves small regions and finer curves for images up to 512 pixels on their longest edge. This improves low-resolution logo lettering but cannot reconstruct missing source detail. Neither AI model is guaranteed to reproduce a complex logo accurately, even when its XML is valid. These changes do not train or fine-tune model weights.

See [STARVECTOR_RESEARCH.md](STARVECTOR_RESEARCH.md) for the investigation and implementation boundary.

## Validate

```sh
cargo test
cargo clippy --all-targets -- -D warnings
node --test web/app.test.cjs
```

## License

Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream notices.
