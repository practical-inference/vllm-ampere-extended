# Boot OOM on Ampere

The engine dies at startup with a CUDA out-of-memory inside KV-cache
initialization — weights loaded fine, then the KV pool allocation fails. On
Ampere the fork has a lever stock vLLM does not: software-fp8 KV cache.

## Symptom

Startup fails shortly after the weights load, with a trace like:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.95 GiB.
GPU 0 has a total capacity of 95.15 GiB of which 16.06 GiB is free.
```

raised from the KV-cache buffer allocation (`torch.zeros`) during engine
startup — the KV-cache initialization, in stock vLLM 0.32.0 serving
Qwen3.8-Flash-Next on a 96 GB A100. The full captured trace:
[recipes/models/Qwen/qwen3.8-flash-next-stock-failure.txt](../recipes/models/Qwen/qwen3.8-flash-next-stock-failure.txt).

## Why it happens

The KV pool scales with `--max-model-len`, and bf16 KV bytes for
MLA-class checkpoints at very long context exceed what is left after the
weights: on the captured failure, bf16 KV at 262144 context needs ~19 GiB
next to ~79 GiB of weights on a 96 GB card. Lowering
`--gpu-memory-utilization` cannot fix this — the pool is already the thing
that does not fit; the levers are fewer KV bytes (dtype) or a shorter
context.

## The fix

Serve with the fork's software-fp8 KV path — on SM80 (no fp8 units) the
encode/dequant runs in Triton kernels, halving KV bytes vs bf16:

```bash
docker run --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v /data/models:/models:ro \
  ghcr.io/practical-inference/vllm-ampere-extended:v2026.10.1-ae.1 \
  --model /models/qwen3.8-flash \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 11000000000 \
  --api-key "$VLLM_API_KEY"
```

- `--kv-cache-dtype fp8` — the software e4m3 layout (see
  [docs/ampere/sparse-attention.md](../docs/ampere/sparse-attention.md)
  for what the backend does with it).
- `--kv-cache-memory-bytes` — pins the KV pool explicitly instead of
  deriving it from `--gpu-memory-utilization`; use the registered spelling
  (the manifest's `--kv-cache-memory` works only as a prefix abbreviation).

This is the mechanism that boots the Qwen3.8-Flash-Next checkpoint stock
vLLM cannot: the fork-vs-stock comparison, the pool math, and the measured
numbers are in
[recipes/models/Qwen/qwen3.8-flash-next-ampere.md](../recipes/models/Qwen/qwen3.8-flash-next-ampere.md).

If fp8 is not an option for your model, the remaining levers are the stock
ones: a smaller `--max-model-len`, or a smaller checkpoint — the Qwen
checkpoint does not fit an 80 GB A100 at any setting (~80 GiB weights
alone).

## Verify

KV-cache init prints the pool size per group — if the numbers are far below
expectations, a flag is wrong. Steps:
[runbooks/run-container.md](run-container.md#verify-it-is-alive).
