# vllm-ampere-extended container image

Drop-in OpenAI-compatible vLLM server for Ampere-class and newer NVIDIA GPUs.
One image carries SASS for compute capabilities 8.0 (A100), 8.6 (RTX 30xx /
A5000-class), 8.9 (RTX 40xx / L40 / Ada), 9.0a (Hopper), 10.0f (Blackwell
sm100/sm103 family) and 12.0 (RTX 50xx); unknown future architectures JIT
from the included PTX instead of failing.

Running it as a long-running service — systemd/shell templates, the
arguments that matter, and per-feature availability by tag:
[runbooks/run-container.md](../runbooks/run-container.md).

## Requirements

- NVIDIA driver >= 580 (the image runs CUDA 13 user-space). Ampere setups on
  535-era drivers are out of scope for this image.
- NVIDIA Container Toolkit (`--gpus all`).

## Run (fail-closed defaults)

```bash
docker run --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/practical-inference/vllm-ampere-extended:<tag> \
  --model /models/<model> --api-key <key>
```

- The documented publish binds `127.0.0.1`. If you publish on `0.0.0.0`
  without `--api-key`, you are offering free GPU compute to your whole
  network — don't.
- `--ipc=host` exists for NCCL shared-memory performance; on multi-tenant
  hosts prefer `--shm-size` after measuring.
- Weights are mounted read-only in the documented line. Keep them ro: models
  loaded with `--trust-remote-code` can write anywhere the container can.

## Tags

- `sha-<7>` — immutable, one per released commit. Pin it (or the `@sha256:`
  digest) for anything you need reproducible.
- `vX.Y.Z-ae.N` — one per release. `latest` floats to the newest release and
  is the only mutable tag.

## Validated configurations

See [docker/ampere-manifest.yaml](ampere-manifest.yaml). Tiers:

- `validated` — full gate suite green on the named architecture.
- `smoke-validated` — boots and generates on the named architecture; no perf
  claim.
- `compiled-untested` — in the binary, no hardware has run it. sm89 and
  newer ship in this state until per-architecture CI buddies report.
