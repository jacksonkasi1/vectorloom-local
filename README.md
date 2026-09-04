# VectorLoom Local

Local-only image vectorization for Apple Silicon. The UI is deliberately simple: upload a PNG, JPG, or WebP; VectorLoom traces it automatically; preview and download the SVG.

## Run

```sh
cargo run --release
open http://127.0.0.1:3000
```

All files are processed in memory by the process bound to `127.0.0.1`; no image or SVG is persisted or transmitted.

## Current inference policy

- `VECTOR_MODEL=8b` is the default declared quality target; `VECTOR_MODEL=1b` is accepted for future benchmarking/fallback work.
- The included runtime uses VTracer's Rust library with automatically selected palette size, spline fitting, seam-free cutout compositing, and SVG optimization.
- The API clearly reports the active engine and will not claim StarVector or Metal execution when those components are not linked.
- The checked Rust reference supports safetensors and GGUF on CPU/CUDA, but does not currently expose a Metal feature. The next phase is a validated Candle Metal port for the exact 1B and 8B model profiles.

See [STARVECTOR_RESEARCH.md](STARVECTOR_RESEARCH.md) for the investigation and implementation boundary.

## Validate

```sh
cargo test
cargo clippy --all-targets -- -D warnings
```

## License

Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream notices.
