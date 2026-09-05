# Hosted deployment

The Docker image is a full public VectorLoom service: it serves the browser UI
and the conversion API from one URL. Once its container starts, it serves the UI, then
downloads either missing StarVector checkpoint into mounted `/models` storage
in the background. The model selector shows live preparation progress.

Both models need approximately 20 GB of persistent storage. The configuration
prefers an L40S (48 GB VRAM), with A100 80 GB, H100, and RTX PRO 6000 capacity
fallbacks. Only the most recently used model is held in GPU memory.

## Modal

Install and authenticate the CLI, then deploy from the repository root:

```sh
python -m pip install modal
modal setup
modal deploy deploy/modal_deploy.py
```

The command prints the public `modal.run` URL. The named `vectorloom-models`
volume retains checkpoints across cold starts and deployments.

The web page defaults to direct tracing, so logo uploads do not wait for AI
inference. Both installed AI models remain selectable. A warm worker reuses
the same model between requests; switching models or a cold start reloads it.
Modal is configured to scale down after 120 idle seconds, so a first visit can
still incur a container/GPU cold start. Checkpoint persistence does not eliminate
that startup time.

## API

Send multipart fields `image` and `model` (`trace`, `1b`, or `8b`) to
`POST /api/vectorize/jobs`. The response contains `job_id`. Poll
`GET /api/vectorize/jobs/{job_id}` until `state` is `complete` (with `result`)
or `failed` (with `error`). `POST /api/vectorize` is also available for synchronous
clients, but AI generation can exceed an HTTP proxy's request timeout.

Jobs are in-memory and the deployment uses one container so polls reach the
owning process. A deployment/restart discards jobs; submit again if the API
reports that a job no longer exists.

## Beam

Beam's current Python endpoint product is designed around a Python request
handler rather than exposing an arbitrary HTTP server port. The Docker image is
ready for Beam's custom-image build flow, but its public UI deployment requires
their container-server offering or a small Python HTTP gateway. Do not deploy
the raw endpoint until that gateway is in place: it would expose only an API,
not this HTML application.
