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

## Format

Each recipe: hardware, exact configuration, mechanism, data table vs
alternatives, and reproduction commands. Benchmarks are run on the same
hardware class the recipe targets; see each recipe for its setup. For
compatibility-creating backends (models that cannot run on this hardware
otherwise — there is no dense-attention fallback for the sparse-attention
model classes), the data is absolute serving numbers plus the kernel-level
analysis, not a comparison against a fallback that doesn't exist.

## Contributing

Ran a config comparison on Ampere or older hardware? Open a PR adding a
recipe — include the raw numbers and how you measured them. Negative results
(a config that loses) are welcome; they save everyone else the experiment.
