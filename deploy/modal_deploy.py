"""Deploy VectorLoom's complete web UI to Modal.

Run:  pip install modal && modal deploy deploy/modal_deploy.py
The first deployment downloads about 20 GB of public StarVector checkpoints
into the persistent `vectorloom-models` volume before the site becomes ready.
"""
import os
import subprocess
import time
import urllib.request

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
    # Jobs live in the web process while they run. Keep a single worker so
    # polling is always routed back to the worker that owns the job.
    max_containers=1,
)
@modal.concurrent(max_inputs=16)
@modal.web_server(3000, startup_timeout=1800)
def vectorloom():
    # Serve the UI immediately; the Rust service downloads missing checkpoints
    # in the background and exposes their live progress through /api/models.
    os.environ["VECTOR_AUTO_DOWNLOAD"] = "all"
    os.environ["VECTOR_MODEL"] = "8b"
    os.environ["VECTOR_OFFICIAL_8B_RUNTIME"] = "1"
    subprocess.Popen(["/app/vectorloom-local"])


@app.function(
    image=image,
    gpu=["L40S", "A100-80GB", "H100", "RTX-PRO-6000"],
    volumes={"/models": models},
    timeout=60 * 60,
)
def quality_probe():
    """Run the supplied badge image entirely in Modal and preserve its response."""
    os.environ["VECTOR_AUTO_DOWNLOAD"] = "all"
    os.environ["VECTOR_MODEL"] = "8b"
    os.environ["VECTOR_OFFICIAL_8B_RUNTIME"] = "1"
    os.environ["VECTOR_DEBUG_RAW_OUTPUT"] = "/models/probe/raw-starvector-output.txt"
    service = subprocess.Popen(["/app/vectorloom-local"])
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen("http://127.0.0.1:3000/api/health", timeout=2)
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("VectorLoom did not start")
        image = open("/models/probe/favicon_new.png", "rb").read()
        boundary = "VectorLoomProbeBoundary"
        body = b"".join([
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"favicon_new.png\"\r\nContent-Type: image/png\r\n\r\n".encode(),
            image,
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n8b\r\n--{boundary}--\r\n".encode(),
        ])
        request = urllib.request.Request(
            "http://127.0.0.1:3000/api/vectorize",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        response = urllib.request.urlopen(request, timeout=3600).read()
        os.makedirs("/models/probe", exist_ok=True)
        open("/models/probe/result.json", "wb").write(response)
        models.commit()
        return {"bytes": len(response)}
    finally:
        service.terminate()
