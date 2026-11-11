# Run the container

Copy-paste templates for serving `ghcr.io/practical-inference/vllm-ampere-extended` as a long-running service, plus the arguments that actually matter — including the ones our delta adds.

For what the image contains and its tag policy, see [docker/ampere-image.md](../docker/ampere-image.md). For per-model validated configs, see [docker/ampere-manifest.yaml](../docker/ampere-manifest.yaml) and [recipes/](../recipes/).

## Quickstart

```bash
docker run --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v /data/models:/models:ro \
  ghcr.io/practical-inference/vllm-ampere-extended:v2026.10.1-ae.1 \
  --model /models/qwen3.8-flash --api-key "$VLLM_API_KEY"
```

Then check it is alive:

```bash
curl -s http://127.0.0.1:8000/health && echo OK
```

The publish binds `127.0.0.1`. If you publish on `0.0.0.0` without `--api-key`, you are offering free GPU compute to your whole network — don't.

## Which tag

| Tag | Use it for | Mutability |
| --- | --- | --- |
| `vX.Y.Z-ae.N` | stable serving; matches a GitHub release | immutable |
| `nightly-YYYYMMDD` | tracking the nightly integrate | immutable |
| `sha-<7>` (or `@sha256:` digest) | reproducing an exact build | immutable |
| `latest` | convenience | **floats** — never pin production to it |

Pin an immutable tag (or digest) everywhere a machine restarts unattended.

Spelling note: git release tags and GHCR container tags can spell the same version differently (`v2026.10.1+ae.1` as a git tag, `v2026.10.1-ae.1` on GHCR) — always copy the tag string from the registry you are pulling from.

Which features your tag carries: [docs/ampere/availability.md](../docs/ampere/availability.md) — generated each release from `docs/ampere/features.list`; the "First available in" column below summarizes, the generated page is canonical.

## systemd unit

`/etc/systemd/system/vllm-ampere.service`:

```ini
[Unit]
Description=vLLM Ampere-extended server
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=exec
EnvironmentFile=/etc/vllm-ampere.env
ExecStart=/usr/bin/docker run --rm --name vllm-ampere \
  --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  --env-file /etc/vllm-ampere.env \
  -v /data/models:/models:ro \
  ghcr.io/practical-inference/vllm-ampere-extended:v2026.10.1-ae.1 \
  --model /models/qwen3.8-flash \
  --max-model-len 262144 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 11000000000 \
  --gpu-memory-utilization 0.965 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --max-cudagraph-capture-size 32 \
  --no-scheduler-reserve-full-isl \
  --speculative-config '{"method":"qwen4_exp_mtp","num_speculative_tokens":2}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --api-key ${VLLM_API_KEY}
ExecStop=/usr/bin/docker stop vllm-ampere
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`/etc/vllm-ampere.env` (mode 600, root-owned — it holds the API key):

```bash
VLLM_API_KEY=sk-change-me
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-ampere
journalctl -u vllm-ampere -f          # server logs
```

The launch flags above are the **validated qwen3.8-flash configuration** from [docker/ampere-manifest.yaml](../docker/ampere-manifest.yaml) (SM80, tier-1 green) — the manifest's `launch:` lines are bare-metal form starting with `vllm serve`; inside the container the entrypoint supplies `vllm serve`, so you pass only the flags. One spelling note: the manifest writes `--kv-cache-memory` (accepted as a prefix abbreviation of the registered `--kv-cache-memory-bytes`); the templates use the registered spelling — prefer it, abbreviations break when another flag with the same prefix lands. Swap the model line for your model's manifest entry — do not invent flags; the manifest entries are the ones we run.

## Shell-script alternative (no systemd)

```bash
#!/usr/bin/env bash
set -euo pipefail
source /etc/vllm-ampere.env
exec docker run --rm --name vllm-ampere \
  --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  --env-file /etc/vllm-ampere.env \
  -v /data/models:/models:ro \
  ghcr.io/practical-inference/vllm-ampere-extended:v2026.10.1-ae.1 \
  --model /models/qwen3.8-flash \
  --max-model-len 262144 \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 11000000000 \
  --gpu-memory-utilization 0.965 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --max-cudagraph-capture-size 32 \
  --no-scheduler-reserve-full-isl \
  --speculative-config '{"method":"qwen4_exp_mtp","num_speculative_tokens":2}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --api-key "$VLLM_API_KEY"
```

Run it under your supervisor of choice (tmux, s6, supervisord). `--rm` keeps stopped containers from accumulating; logs go to stdout — capture them.

## Arguments that matter

### Stock vLLM arguments (from the validated manifest entries)

| Argument | Why it matters here |
| --- | --- |
| `--max-model-len` | Long contexts are the point of this fork — but the KV pool scales with it. Start from the manifest value for your model. |
| `--kv-cache-dtype fp8` | Halves the KV pool; the sparse-attention paths are validated with fp8 cache on Ampere. |
| `--gpu-memory-utilization` | The validated configs run high (0.965) — weights + KV must fit; lower it before raising `--max-model-len`. |
| `--max-num-seqs` | Concurrency cap. The spec-tier table (below) keys on batch size, so this interacts with speculative decoding. |
| `--max-cudagraph-capture-size` | Match to `--max-num-seqs` to avoid re-capture stalls at the cap. |
| `--speculative-config` | MTP draft config; see the model's manifest entry for the validated method and draft length. |

### Our delta's configuration

| Setting | What it does | When to set it | First available in |
| --- | --- | --- | --- |
| `TRITON_MLA_SPARSE` backend | Sparse-MLA attention on SM80/SM121 — required for DeepSeek-class and GLM sparse models; selected **automatically** for MLA models on Ampere. Force it with `VLLM_ATTENTION_BACKEND=TRITON_MLA_SPARSE`. | Nothing to do for supported models; the override is for experiments. | `v2026.10.1-ae.1` |
| `num_speculative_tokens_per_batch_size` (inside `--speculative-config`) | Dynamic spec tier: `[[start_bs, end_bs, K], ...]` — draft K tokens per step depending on concurrency. Example: `[[1,8,2],[9,16,1],[17,32,0]]` = K2 below 9 concurrent, K1 to 16, K0 above. | Serving MTP models at varying concurrency. Full semantics: [docs/features/speculative_decoding/dynamic_speculative_decoding.md](../docs/features/speculative_decoding/dynamic_speculative_decoding.md). Not compatible with `--data-parallel-size > 1`. | `v2026.10.1-ae.1` |
| `VLLM_MAMBA_PRICED_ADMISSION=1` | Hybrid-mamba models: when a request cannot admit at its tier's state footprint, admit it at the K=0 tier (speculative window as null row slots) instead of stalling; promotes back when the batch drains. | Hybrid mamba + MTP under load (GLM-5.3-Flash class). Requires a spec-tier table. | **not in any released image yet** — first image builds from the `v2026.10.2.dev135-ae.1` tag (publish pending) |
| `VLLM_MAMBA_K0_SLOT_RELEASE=1` | Hybrid-mamba models: release running rows' idle speculative windows back to the KV pool at the K=0 crossing; refills on promotion. This is what makes the concurrency-32 crossing into the K0 tier pay (+110% at that crossing in our A/B — measured A/B and crash transcripts in upstream PR [#66169](https://github.com/vllm-project/vllm/pull/66169), frozen copies in the [evidence archive](https://github.com/practical-inference/vllm-ampere-extended/releases/download/k0-slot-release-20261106/evidence-k0-slot-release-20261106.tar.gz)). | Same as above; pairs with priced admission. Requires a spec-tier table. | **not in any released image yet** — first image builds from the `v2026.10.2.dev135-ae.1` tag (publish pending) |
| `--kv-cache-valve-threshold <fraction>` | The KV valve: preempts the most-recently-scheduled running request to restore slack before allocation failure. Primary trigger is admission demand (a waiting request stalled on a block shortage one preemption relieves); the fraction adds an earlier proactive trigger. Default `0.0` = off. | With the mamba mechanisms on under pressure, this is part of the recommended configuration — without it the release arm can exhaust the shared pool mid-call (`get_new_blocks` ValueError; repro transcript in the same evidence archive). | **not in any released image yet** — first image builds from the `v2026.10.2.dev135-ae.1` tag (publish pending) |

### Worked example — hybrid mamba under pressure (GLM-5.3-Flash class)

```bash
docker run --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v /data/models:/models:ro \
  --env-file /etc/vllm-ampere.env \
  ghcr.io/practical-inference/vllm-ampere-extended:<next-nightly-tag> \
  --model /models/glm5.3-flash \
  --max-model-len 393216 \
  --speculative-config '{"method":"<mtp-method>","num_speculative_tokens":2,"num_speculative_tokens_per_batch_size":[[1,8,2],[9,16,1],[17,32,0]]}' \
  -e VLLM_MAMBA_PRICED_ADMISSION=1 \
  -e VLLM_MAMBA_K0_SLOT_RELEASE=1 \
  --kv-cache-valve-threshold 0.9 \
  --api-key "$VLLM_API_KEY"
```

The shape above is the pattern validated in our hybrid-mamba A/B (the +110% crossing result); the GLM-5.3-Flash manifest entry is `pending` full validation, so treat the exact MTP method and table as starting points and check [docker/ampere-manifest.yaml](../docker/ampere-manifest.yaml) for the current status per model.

## Verify it is alive

```bash
curl -s http://127.0.0.1:8000/health            # 200 when the engine is up
curl -s http://127.0.0.1:8000/v1/models         # lists the served model
docker logs vllm-ampere --tail 100              # startup + KV-cache init lines
```

KV-cache init prints the pool size per group — if the numbers are far below the manifest's expectations, a flag is wrong (usually `--gpu-memory-utilization` or `--max-model-len`).

## Where numbers come from

Every performance claim in this repo traces to a captured transcript: model pages under [recipes/](../recipes/), image validation in [docker/ampere-manifest.yaml](../docker/ampere-manifest.yaml) (tiers: `validated` / `smoke-validated` / `pending` — see [docker/ampere-image.md](../docker/ampere-image.md) for the full taxonomy including `compiled-untested`), evidence archives attached to each GitHub release.
