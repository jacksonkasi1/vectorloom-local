# Hosted deployment

The Docker image is a full public VectorLoom service: it serves the browser UI
and `POST /api/vectorize` from one URL. It starts only after both 1B and 8B
StarVector checkpoint sets exist in the mounted `/models` directory.

Both models need approximately 20 GB of persistent storage. Use an NVIDIA GPU
with at least 24 GB VRAM for the 1B model; an A100 80 GB is configured because
it can run either model reliably.

## Modal

Install and authenticate the CLI, then deploy from the repository root:

```sh
python -m pip install modal
modal setup
modal deploy deploy/modal_deploy.py
```

The command prints the public `modal.run` URL. The named `vectorloom-models`
volume retains checkpoints across cold starts and deployments.

## Beam

Beam's current Python endpoint product is designed around a Python request
handler rather than exposing an arbitrary HTTP server port. The Docker image is
ready for Beam's custom-image build flow, but its public UI deployment requires
their container-server offering or a small Python HTTP gateway. Do not deploy
the raw endpoint until that gateway is in place: it would expose only an API,
not this HTML application.
