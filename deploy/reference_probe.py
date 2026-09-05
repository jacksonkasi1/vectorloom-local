"""Run an upstream control image using an already-built deployment image.

Set VECTOR_PROBE_IMAGE_ID to a verified Modal image ID and
VECTOR_PROBE_INPUT to a local test PNG before invoking with modal run --detach.
This mounts the current Python adapter without rebuilding the CUDA image.
"""
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import modal

app = modal.App("vectorloom-reference-probe")
image = (modal.Image.from_id(os.environ["VECTOR_PROBE_IMAGE_ID"])
         .add_local_file(str(Path(__file__).resolve().parents[1] / "reference_vectorize.py"), "/probe/reference.py")
         .add_local_file(os.environ["VECTOR_PROBE_INPUT"], "/probe/input.png")) if modal.is_local() else None
volume = modal.Volume.from_name("vectorloom-models")


@app.function(image=image, gpu=["L40S", "A100-80GB", "H100", "RTX-PRO-6000"],
              volumes={"/models": volume}, timeout=1800)
def run():
    directory = Path("/models/probe/runs") / uuid.uuid4().hex
    directory.mkdir(parents=True)
    manifest = {"state": "running", "started_at": time.time()}
    def save():
        (directory / "manifest.json").write_text(json.dumps(manifest))
        volume.commit()
    save()
    print(f"Reference probe results: {directory}", flush=True)
    env = dict(os.environ, VECTOR_DEBUG_RAW_OUTPUT=str(directory / "raw.svg"))
    try:
        with (directory / "runtime.log").open("w") as log:
            process = subprocess.run([
                "/usr/bin/python3", "/probe/reference.py", "/probe/input.png",
                str(directory / "raw.svg"), "/models/starvector-8b-im2svg",
            ], env=env, stdout=log, stderr=log, timeout=1700)
        manifest.update(state="completed" if process.returncode == 0 else "failed",
                        returncode=process.returncode)
    except Exception as error:
        manifest.update(state="failed", error=str(error))
        raise
    finally:
        manifest["finished_at"] = time.time()
        save()
    return {"result_dir": str(directory), **manifest}
