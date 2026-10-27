# GLM-5.3 (SplitQuant) on 6× A100 (TP6) — production-validated, pre-reorg fleet

Hardware: 6× 96GB A100 (SM80), TP6 (`CUDA_DEVICES=0,1,3,4,5,6`), one engine
serving live traffic — the 2026-10-23 fleet, before the 2026-10-25 GPU-loss
reorganization. Checkpoint: GLM-5.3-SplitQuant (served under the name
`glm-5.3-flash`). Measured 2026-10-23 by the two-stage production canary
harness against the live endpoint — these are not lab numbers; the exact
build that produced them is the build in production at the time
(`0.32.1rc3.dev229+gac4573115`, release nightly-20261023).

**NOTE (2026-10-26):** the current production unit is `vllm-glm53-flash`
(GLM-5.3-Flash W4A16+int4-RTN-draft, TP4 on GPUs 0,2,4,5) — a different,
smaller configuration. The sweep below has NOT been re-run against it; the
numbers here describe the 6-GPU SplitQuant deployment that existed on
2026-10-23.

**MTP speculative decoding (N=1) was ACTIVE during these measurements** —
it is permanent unit config (`MTP_N=1`, serving with MTP since 2026-10-01).
The banked A/B from that arm (GLM-5.3-SplitQuant TP2xPP3, ~79% acceptance,
`results/mtp_nightly_20261001`): MTP is worth **+46% / +36% / +19% / +7%**
at c1/c4/c16/c32 vs no-MTP — so the true no-MTP floor at c1 is roughly
~105 tok/s, not the 187.5 measured here. A same-config no-MTP arm on the
current TP4 stack has not been run; N=2 remains untested on this config
(the 2026-10-01 Flash study measured N=2 at +56% c1 on a 16k-context
W4A16 variant).

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
