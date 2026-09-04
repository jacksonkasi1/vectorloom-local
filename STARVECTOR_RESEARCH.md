# StarVector investigation

Completed before implementation on 2026-09-04.

| Area | Finding | Decision |
| --- | --- | --- |
| Upstream | `joanrod/star-vector` is Apache-2.0 and provides Python/Transformers inference. | Do not copy its Python implementation into this Rust app. |
| 1B architecture | CLIP encoder + BatchNorm adapter + BigCode decoder. | Requires a separate typed loader and generation path. |
| 8B architecture | SigLIP encoder + LayerNorm adapter + StarCoder2 decoder. | Treat as a distinct profile; it is the default future quality engine. |
| Model assets | Hugging Face safetensors checkpoints; Rust reference also supports GGUF. | Support both only after parity tests validate output. |
| Rust reference | `oxide-lab/starvector-rs`, Apache-2.0, CPU/CUDA feature set. | Useful design reference, not copied or linked yet. |
| Metal | The inspected reference declares no Metal feature; a working Apple GPU path cannot be asserted. | Report Metal as preferred-but-unlinked until Candle operators and parity are validated. |
| Current local output | VTracer 1.0 Rust library, MIT OR Apache-2.0, supports color segmentation, splines, cutout compositing, and optimized SVG. | Ship as transparent high-quality automatic fallback. |

## Metal port gate

Before enabling `StarVector 8B / Metal / BF16` in production, implement and test:

1. 8B SigLIP vision encoder and LayerNorm adapter on Candle Metal.
2. StarCoder2 KV-cache generation and all required attention/RoPE/index operations on Metal.
3. Safetensors and GGUF loader coverage for both model profiles.
4. CPU-versus-Metal deterministic smoke tests and visual regression fixtures.
5. Backend telemetry for actual device, precision, model load time, inference duration, and peak-memory estimate.

The model is especially suitable for icons, logotypes, diagrams, charts, and SVG-like assets; upstream explicitly warns that it was not trained for natural photographs and illustrations.
