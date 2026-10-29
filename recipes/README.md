<!-- markdownlint-disable MD060 -->

# Recipes

Configs for model + GPU combinations, with the data showing why they beat the
alternatives. Every recipe states the mechanism (why it works) and includes
measured numbers — a config without data is a guess, not a recipe.

**The bar:** we publish a config only when it measurably improves performance
or enables something that otherwise doesn't work. Tuning experiments with
inconclusive results stay unpublished — signal, not noise.

## Index

| Recipe | Hardware | Verdict |
|---|---|---|
| [KV offloading on Ampere](kv-offload-ampere.md) | SM80 (Ampere), long-context doc-summary workload | **Avoid** — connector overhead causes the preemptions it exists to rescue |
| [Qwen3.8-Flash-Next on one A100](models/Qwen/qwen3.8-flash-next-ampere.md) | 1× A100 80 GB (SM80), TP1 | **Runs** — 596 tok/s c1, 1,164 tok/s c64; stock vLLM fails to boot on the same card |
| [GLM-5.3-Flash on 4× A100](models/GLM/glm-5.3-flash-tp4-ampere.md) | 4× A100 80 GB (TP4), production | **Production-validated** — 141–354 tok/s decode with MTP, 128k TTFT 47 s cold → 1 s cached |

## Format

Each recipe: hardware, exact configuration, mechanism, data table vs
alternatives, and reproduction commands. Benchmarks are run on the same
hardware class the recipe targets; see each recipe for its setup. For
compatibility-creating backends (models that cannot run on this hardware
otherwise — there is no dense-attention fallback for the sparse-attention
model classes), the data is absolute serving numbers plus the kernel-level
analysis, not a comparison against a fallback that doesn't exist.

Benchmarks use `recipes/bench/serve_matrix.py` (fixed sampling, fully echoed
configuration); `recipes/bench/field_test.sh` wraps it into a one-command
reproduction whose output pastes straight into an issue.

## Contributing

Ran a config comparison on Ampere or older hardware? Open a PR adding a
recipe — include the raw numbers and how you measured them. Negative results
(a config that loses) are welcome; they save everyone else the experiment.
