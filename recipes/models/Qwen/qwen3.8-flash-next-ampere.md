# Qwen3.8-Flash-Next (Qwen4Exp) on one A100 — SM80

Hardware: single 96GB A100 (SM80, 400W fleet card), TP1/PP1, one engine.
Checkpoint: `Qwen3.8-Flash-Next-AWQ-g32` (compressed-tensors AWQ g32 asym +
BF16 PLE n-gram table, 38 shards). Measured 2026-10-26.

## Numbers

| Scenario | Config | Result |
|---|---|---|
| Single-stream decode | 500 in / 256 out, temp 0, c=1 | **596 tok/s**, TTFT 1.41 s |
| 64-concurrent | 500 in / 256 out, c=64 | **1,164 tok/s aggregate**, per-stream 254 tok/s, TTFT p50 10.2 s |
| Long-context prefill | 30,068 in / 256 out, c=4 | TTFT **6.35 s** |

Source of truth: `recipes/bench/serve_matrix.py` JSON (config block echoes
the full invocation). Harness: `recipes/bench/serve_matrix.py`.

## Mechanism

Why a stock-vs-fork comparison is not a zero baseline here: Qwen4Exp's
sparse-indexer architecture has no dense fallback — but upstream ships the
NVIDIA backend (`qwen4_exp/nvidia/`) that can boot it on SM80 with the PLE
n-gram table offloaded to pinned host RAM (`VLLM_PLE_CPU_OFFLOAD=1`, ~52 GiB
that cannot fit next to the ~80 GiB weights). What stock does NOT ship for
SM80 is what this fork carries:

- **fp8 KV cache on SM80** — upstream gates fp8 KV to fp8-hardware
  generations; the fork's software e4m3 emulation path doubles the KV pool
  vs bf16 (~500K tokens vs ~250K at these settings).
- **MTP speculative decoding (K=2)** with the Qwen4Exp native drafter —
  the single-stream 596 tok/s is MTP-amplified; stock serves this model
  without a working spec-decode config on this stack.
- **KV offload connector + mamba shared-prefix checkpointing** — cached
  agentic turns skip the full-prompt mamba replay.

## Stock upstream vLLM on the same hardware (measured 2026-10-26)

Stock upstream **vLLM 0.32.0** (current stable, fresh venv, minimal flags,
`VLLM_PLE_CPU_OFFLOAD` set — upstream warns it is now an unknown env, but
resolves `EngramConfig(cpu_offload=True)` regardless): weights load, then
the engine **fails to start**:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.95 GiB.
GPU 0 has a total capacity of 95.15 GiB of which 16.06 GiB is free.
... v1/engine/core.py:315, in _initialize_kv_caches
```

bf16 KV for this architecture at 262144 context does not fit the ~15 GiB
left after the ~79 GiB of weights on a 96 GB card. The fork boots the same
checkpoint because its SM80 software-fp8 KV path halves KV bytes and pins
the pool explicitly (`--kv-cache-dtype fp8 --kv-cache-memory 11G`). Capture
procedure: `recipes/bench/capture_stock_failure.sh` (dev-slot window; stops
the fork unit, serves stock with fork-only features stripped, captures the
outcome, restores). Raw evidence in this directory:
`qwen3.8-flash-next-stock-failure.txt`, `qwen3.8-flash-next-bench-fork.json`
(full config echo + numbers), `qwen3.8-flash-next-demo.gif` (token-arrival
replay, stock panel shows the failure).

For an upstream 0.29-era data point, stock could boot this checkpoint with
PLE offload and a halved bf16 pool — current stable cannot. Either way the
fork's numbers above are what this hardware serves today.

## Reproduce

```bash
# fork side (any release >= nightly-20261023)
vllm serve /data/models/cyankiwi --served-model-name qwen3.8-flash-next \
  --dtype bfloat16 --kv-cache-dtype fp8 --max-model-len 262144 ...
# benchmark
python recipes/bench/serve_matrix.py --base-url http://localhost:8017/v1 \
  --model qwen3.8-flash-next --out results.json
```

Exact production serve arguments: `serve-leoncca-tp1.sh` (fleet repo).
