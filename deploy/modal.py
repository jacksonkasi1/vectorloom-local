"""Deploy VectorLoom's complete web UI to Modal.

Run:  pip install modal && modal deploy deploy/modal.py
The first deployment downloads about 20 GB of public StarVector checkpoints
into the persistent `vectorloom-models` volume before the site becomes ready.
"""
import subprocess

import modal

app = modal.App("vectorloom")
image = modal.Image.from_dockerfile("Dockerfile", context_dir=".", add_python="3.12")
models = modal.Volume.from_name("vectorloom-models", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/models": models},
    timeout=60 * 60,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=1)
@modal.web_server(3000, startup_timeout=1800)
def vectorloom():
    # Persist any first-run checkpoint downloads before accepting browser traffic.
    subprocess.run(["/app/vectorloom-local", "--bootstrap-models"], check=True)
    models.commit()
    subprocess.Popen(["/app/vectorloom-local"])
