<!-- markdownlint-disable MD060 -->

# Recipes

The site, the command builder, and the JSON API are built by
[vllm-project/recipes](https://github.com/vllm-project/recipes). That
repository owns the schema and the shared vocabulary (hardware profiles,
serving strategies, KV-offload deployments) and pins this directory to a
commit. Omni (image, video, audio) recipes live in vllm-project/vllm-omni, and
Ascend NPU content lives in vllm-project/vllm-ascend as overlays that merge into
these recipes when the site is built. Don't add Ascend hardware here: open those
changes against `recipes/overlays/` in vllm-ascend.

Configs for model + GPU combinations, with the data showing why they beat the
alternatives. Every recipe states the mechanism (why it works) and includes
measured numbers — a config without data is a guess, not a recipe.

**The bar:** we publish a config only when it measurably improves performance
or enables something that otherwise doesn't work. Tuning experiments with
inconclusive results stay unpublished — signal, not noise.

Don't set `--gpu-memory-utilization` for CUDA or ROCm hardware, even when a
model card recommends a value. There, vLLM sizes the KV cache from the memory
measured after warmup and defaults the setting to 1.0, so a lower value only
shrinks the KV cache. Set it only to leave room for something vLLM doesn't
measure, such as another process sharing the GPU or the CPU on a unified-memory
device like DGX Spark; see
[GPU Memory Utilization](https://docs.vllm.ai/en/latest/configuration/optimization/#gpu-memory-utilization).
Values for other hardware (TPU, XPU, CPU) still size the KV cache there.

Follow the recipes repository's
[add-recipe skill](https://github.com/vllm-project/recipes/blob/main/.agents/skills/add-recipe/SKILL.md)
for the schema and the validation steps. To validate a change exactly as this
repository's `Recipes` PR check does, run from a checkout of
`vllm-project/recipes`:

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
