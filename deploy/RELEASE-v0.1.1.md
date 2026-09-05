# VectorLoom v0.1.1 — macOS

Universal app for Apple Silicon and Intel Macs.

- Direct tracing is the default, with finer curves and tiny regions preserved for small logos.
- Image uploads use asynchronous jobs with progress and connection retries.
- Native image picker and SVG save panel support the embedded Mac interface.
- Model download controls are available in the desktop app. Models are optional for tracing and download only when requested.
- Desktop model administration accepts requests only from the local app origin.

Download the DMG, drag VectorLoom to Applications, and replace the previous app. Existing downloaded checkpoints are retained. A ZIP is also provided.

This build is ad-hoc signed, **not Apple-notarized**. macOS may require approval in Privacy & Security on first launch.

Direct tracing is recommended for logos and lettering. AI output can still distort details; no model training is included. The desktop app uses its native Rust/Metal/CPU backend. The hosted Transformers/CUDA runtime is a separate server deployment, not bundled in the Mac app.

Validation: Rust and JavaScript tests, strict bundle signature verification, both binary architectures, DMG integrity, and a logo conversion using the packaged backend. Apple Silicon execution and native dialogs were not interactively tested on this Intel build host.
