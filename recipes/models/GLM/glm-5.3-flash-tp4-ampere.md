# GLM-5.3-Flash on 4× A100 (TP4) — production-validated

Hardware: 4× 96GB A100 (SM80), TP4, one engine serving live traffic.
Checkpoint: GLM-5.3-Flash W4A16+int4-RTN-draft (MoE, sparse-attention
class). Measured 2026-10-23 by the two-stage production canary harness
against the live endpoint (`vllm-glm53-flash`, then on physical GPUs
0,4,5,6 — the 2026-10-25 reorg moved the unit to GPUs 0,2,4,5; same model
class and config family) — these are not lab numbers; the exact build that
produced them is the build in production at the time
(`0.32.1rc3.dev229+gac4573115`, release nightly-20261023).

**MTP speculative decoding (N=2) was ACTIVE during these measurements.**
The serving boot's own config echo (service log 2026-10-23 08:22:49):
`speculative_config=SpeculativeConfig(method='mtp',
model='/data/models/glm-5.3-w4a16-mtp', num_spec_tokens=2)`. This matches
the banked same-model A/B (`~/glm53-perf/results/flash_mtp_nightly_20261001`,
Flash W4A16 TP4, 16k ctx): **99.3 tok/s no-MTP → 155.3 with MTP N=2** at
c1 (+56%) — the 187.5 here is the same N=2 config with a matured serving
setup (longer ctx, KV pin). For scale, Big GLM (SplitQuant TP2xPP3) is a
different model class: 39.4 no-MTP → 57.5 with MTP N=1 at c1 (banked
`results/mtp_nightly_20261001`).

## Decode (concurrency sweep, synthetic workloads, temp 0)

| Workload | Concurrency | Decode tok/s |
| --- | --- | --- |
| short greeting (44 tok out) | 1 | 141.5 |
| code few-shot (300 tok out) | 1 | **187.5** |
| agent tool-schema (500 tok out) | 1 | 173.3 |
| long-context pack | 4 | 239.5 |
| long-context pack | 16 | 322.7 |
| long-context pack | 32 | **354.4** |

Raw: `glm-5.3-flash-conc-sweep.{txt,json}` (per-request rows included).

## Long-context prefill (4 random prompts, concurrency 2, chunked prefill)

| Context | TTFT (cold) | TTFT (prefix-cached) |
| --- | --- | --- |
| 100,000 | 35.5 – 40.1 s | — |
| 128,000 | **47.1 s** | 1.0 s |
| 200,000 | 34.7 s | — |

Cold 128k run raw: `glm-5.3-flash-ctx128000-cold.txt` (full
`vllm bench serve` output; warm runs demonstrate prefix caching paying off
at agent-workload scale — 47 s → 1 s on a repeated 128k context).

## Accuracy

| Measurement | Result | Provenance |
| --- | --- | --- |
| GSM8K, 1319 samples @1024 tok | **97.2%** (invalid 0.3%) | dedicated deep run 2026-10-13 against the then-prod build; value recorded in the promote-gate script (`vllm_promote_prod.sh` standing-gate comment); raw JSON not retained, log: `glm_gsm8k1319_maxtok1024_20261013.txt` |

The nightly canary's accuracy gates (GSM8K-1319 96.1%, needle 1.00, run
2026-10-23 06:09–06:24) measure the **qwen** stage-1 endpoint, not GLM —
they are deliberately not reproduced here. GLM's post-publish canary is a
~4-minute tier-0 smoke, not a full accuracy suite.

## Why this is measured here

Sparse-attention MoE serving on SM80 at this scale is a fork capability —
the same TRITON_MLA_SPARSE / software-fp8 stack the recipes index covers,
validated every nightly against live traffic before a tag ships. The canary
harness (gsm8k + needle + cold-context points) runs against production; a
red gate blocks the release rather than shipping it.

## Reproduce

The raw numbers above came from the production canary harness (a
concurrency-sweep client for the decode table; `vllm bench serve` for the
long-context rows — both committed in this directory). To re-measure with
the current harness:

```bash
# full matrix against a running endpoint (single/batch/longctx/demo)
python recipes/bench/serve_matrix.py --base-url http://localhost:8080/v1 \
  --model glm-5.3-flash --out results.json
# long-context prefill rows, as the canary runs them
vllm bench serve --backend openai --base-url http://localhost:8080 \
  --model glm-5.3-flash --dataset-name random --random-input-len 128000 \
  --random-output-len 64 --num-prompts 4 --max-concurrency 2
```
