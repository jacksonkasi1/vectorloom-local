"""Deploy VectorLoom's complete web UI to Modal.

Run:  pip install modal && modal deploy deploy/modal_deploy.py
The first deployment downloads about 20 GB of public StarVector checkpoints
into the persistent `vectorloom-models` volume before the site becomes ready.
"""
import os
import subprocess

import modal

app = modal.App("vectorloom")
image = modal.Image.from_dockerfile("Dockerfile", context_dir=".", add_python="3.12")
models = modal.Volume.from_name("vectorloom-models", create_if_missing=True)


@app.function(
    image=image,
    # The 8B checkpoint exceeds the A10G's VRAM. Prefer the cost-effective
    # 48 GB L40S, with compatible high-memory pools as capacity fallbacks.
    gpu=["L40S", "A100-80GB", "H100", "RTX-PRO-6000"],
    volumes={"/models": models},
    timeout=60 * 60,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=1)
@modal.web_server(3000, startup_timeout=1800)
def vectorloom():
    # Serve the UI immediately; the Rust service downloads missing checkpoints
    # in the background and exposes their live progress through /api/models.
    os.environ["VECTOR_AUTO_DOWNLOAD"] = "all"
    subprocess.Popen(["/app/vectorloom-local"])
