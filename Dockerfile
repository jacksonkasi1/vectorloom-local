# CUDA build image: compile the Rust Candle backend with NVIDIA support.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates clang curl git pkg-config && \
    rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
# Modal's image builder has CUDA tooling but no attached GPU. Compile kernels
# for the deployment GPU explicitly instead of asking nvidia-smi at build time.
# Ampere compute capability covers the A10G used by the hosted runtime.
ENV CUDA_COMPUTE_CAP=80
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY web ./web
RUN cargo build --release --no-default-features --features cuda

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV VECTOR_BIND=0.0.0.0 \
    VECTOR_PORT=3000 \
    VECTOR_MODEL_DIR=/models \
    RUST_LOG=info
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install --no-cache-dir transformers==4.49.0 tokenizers==0.21.1 accelerate pillow sentencepiece omegaconf fairscale matplotlib beautifulsoup4 svgpathtools cairosvg && \
    python3 -m pip install --no-cache-dir --no-deps --ignore-requires-python git+https://github.com/joanrod/star-vector.git
COPY --from=builder /src/target/release/vectorloom-local /app/vectorloom-local
COPY reference_vectorize.py /app/reference_vectorize.py
COPY web /app/web
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh && mkdir -p /models
EXPOSE 3000
# Modal starts its own Python function runner.  It launches the Rust process
# explicitly from deploy/modal_deploy.py, so no container entrypoint is set
# here (an entrypoint would start a competing copy of the server).
