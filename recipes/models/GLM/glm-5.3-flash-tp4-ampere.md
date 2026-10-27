# GLM-5.3-Flash on 4× A100 (TP4) — production-validated

Hardware: 4× 96GB A100 (SM80), TP4, one engine serving live traffic.
Checkpoint: GLM-5.3-Flash W4A16+int4-RTN-draft (MoE, sparse-attention
class). Measured 2026-10-23 by the two-stage production canary harness
against the live endpoint (`vllm-glm53-flash`, then on physical GPUs
0,4,5,6 — the 2026-10-25 reorg moved the unit to GPUs 0,2,4,5; same model
class and config family) — these are not lab numbers; the exact build that
produced them is the build in production at the time
(`0.32.1rc3.dev229+gac4573115`, release nightly-20261023).

**MTP speculative decoding (N=1) was ACTIVE during these measurements** —
permanent unit config, serving since 2026-10-01. Banked same-model A/B
(`~/glm53-perf/results/flash_mtp_nightly_20261001`, Flash W4A16 TP4,
16k ctx, 2026-10-01): **99.3 tok/s no-MTP → 155.3 with MTP N=2** at c1
(+56%); the 187.5 here is the same class at N=1 with a matured config
(longer ctx, KV pin). Do not divide these numbers by another model's
acceptance factor — for scale, Big GLM (SplitQuant TP2xPP3) is a different
model class: 39.4 no-MTP → 57.5 with MTP N=1 at c1 (banked
`results/mtp_nightly_20261001`).

## Decode (concurrency sweep, synthetic workloads, temp 0)

| Workload | Concurrency | Decode tok/s |
|---|---|---|
| short greeting (44 tok out) | 1 | 141.5 |
| code few-shot (300 tok out) | 1 | **187.5** |
| agent tool-schema (500 tok out) | 1 | 173.3 |
| long-context pack | 4 | 239.5 |
| long-context pack | 16 | 322.7 |
| long-context pack | 32 | **354.4** |

Raw: `glm-5.3-flash-conc-sweep.{log,json}` (per-request rows included).

## Long-context prefill (4 random prompts, concurrency 2, chunked prefill)

| Context | TTFT (cold) | TTFT (prefix-cached) |
|---|---|---|
| 100,000 | 35.5 – 40.1 s | — |
| 128,000 | **47.1 s** | 1.0 s |
| 200,000 | 34.7 s | — |

Cold 128k run raw: `glm-5.3-flash-ctx128000-cold.log` (full
`vllm bench serve` output; warm runs demonstrate prefix caching paying off
at agent-workload scale — 47 s → 1 s on a repeated 128k context).

## Accuracy canary (same window, same build)

| Gate | Result |
|---|---|
| GSM8K, 1319 samples @1024 tok | **96.1%** (invalid 0.9%) |
| GSM8K-32 smoke | 96.9% |
| Needle retrieval + drift (min recall) | **1.00** |

## Why this is measured here

Sparse-attention MoE serving on SM80 at this scale is a fork capability —
the same TRITON_MLA_SPARSE / software-fp8 stack the recipes index covers,
validated every nightly against live traffic before a tag ships. The canary
harness (gsm8k + needle + cold-context points) runs against production; a
red gate blocks the release rather than shipping it.

## Reproduce

```bash
# concurrency sweep against a running endpoint
python recipes/bench/serve_matrix.py --base-url http://localhost:8080/v1 \
  --model glm-5.3-flash --out results.json
# long-context prefill (vllm bench serve, as the canary runs it)
vllm bench serve --backend openai --base-url http://localhost:8080 \
  --model glm-5.3-flash --dataset-name random --random-input-len 128000 \
  --random-output-len 64 --num-prompts 4 --max-concurrency 2
```
