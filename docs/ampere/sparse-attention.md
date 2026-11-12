# Sparse attention on Ampere (TRITON_MLA_SPARSE)

`TRITON_MLA_SPARSE` is a pure-Triton sparse-MLA attention backend for SM80
(A100) and SM121 (GB10) — the DeepSeek sparse-attention architecture
(Lightning Indexer, paged-MQA logits kernels) on hardware upstream's sparse
paths do not reach. For these model classes there is no dense fallback:
sparse attention is part of the trained architecture, so either it works on
your hardware or the model does not run.

## Selection and configuration

Nothing to set for supported models: on SM80/SM121 the backend is selected
automatically for sparse-MLA models (DeepSeek-class, GLM5Next, Qwen4Exp).
Upstream's other sparse backends gate to SM90+/SM100 hardware, so on SM80
this is the only sparse-MLA backend that runs; on SM121 with fp8 KV cache
the FlashInfer SM120 sparse backend is preferred when eligible and selection
falls through to this one otherwise.

| Setting | What it does | When to set it |
| --- | --- | --- |
| — (default) | Auto-selected for sparse-MLA models on SM80/SM121 | Nothing to do |
| `VLLM_ATTENTION_BACKEND=TRITON_MLA_SPARSE` | Forces the backend, skipping auto-selection | Experiments only — a wrong force fails at startup with the backend's rejection reason |
| `--kv-cache-dtype fp8` | Software e4m3 KV cache (alias for the `fp8_ds_mla` layout): encode and dequant run in Triton kernels on cards without fp8 units (pre-SM89). Halves the KV bytes vs bf16 | Long-context serving on SM80 — measured pool sizes in the [Qwen3.8-Flash-Next recipe](../../recipes/models/Qwen/qwen3.8-flash-next-ampere.md) |

Two shape constraints from the backend:

- The KV block size must be a multiple of 64 — the DSA indexer shares the
  KV-cache group and requires 64. Auto-selection settles on 64; larger
  multiples (e.g. 128) work and lower profile-time peak memory for very
  long contexts.
- Rope-free ("NoPE") checkpoints run the 512-wide head shape and accept
  bf16/fp16 and the `fp8_ds_mla` (`fp8`) cache dtypes; standard MLA
  checkpoints (576-wide) are unrestricted.

## Worked example

Serve the validated Qwen3.8-Flash-Next configuration (sparse-indexer
architecture, SM80) — the backend needs no flag:

```bash
docker run --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v /data/models:/models:ro \
  ghcr.io/practical-inference/vllm-ampere-extended:v2026.10.1-ae.1 \
  --model /models/qwen3.8-flash \
  --kv-cache-dtype fp8 \
  --api-key "$VLLM_API_KEY"
```

The full validated launch line (context length, spec-decode, memory flags)
is the `qwen3.8-flash` entry in [docker/ampere-manifest.yaml](../../docker/ampere-manifest.yaml);
service templates live in [runbooks/run-container.md](../../runbooks/run-container.md).

## Evidence

- [recipes/models/Qwen/qwen3.8-flash-next-ampere.md](../../recipes/models/Qwen/qwen3.8-flash-next-ampere.md)
  — Qwen4Exp sparse-indexer on one A100, including the stock-vLLM boot
  failure on the same checkpoint.
- [recipes/models/GLM/glm-5.3-flash-tp4-ampere.md](../../recipes/models/GLM/glm-5.3-flash-tp4-ampere.md)
  — sparse-attention MoE on 4× A100 (TP4).
- Per-tag availability: [docs/ampere/availability.md](availability.md)
  (first release: `v2026.10.1+ae.1`).
