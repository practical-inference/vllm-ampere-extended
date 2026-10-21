<!-- markdownlint-disable MD001 MD041 MD060 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">The latest vLLM models on the GPUs people actually own</h3>

<p align="center">
  <a href="https://github.com/practical-inference/vllm-ampere-extended/releases"><img alt="Latest release" src="https://img.shields.io/github/v/release/practical-inference/vllm-ampere-extended?include_prereleases"></a>
  <a href="https://github.com/practical-inference/vllm-ampere-extended/issues"><img alt="Issues" src="https://img.shields.io/github/issues/practical-inference/vllm-ampere-extended"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-13.2-green">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-lightgrey">
</p>

---

## Why this exists

Upstream vLLM targets modern datacenter GPUs — excellent hardware, and out of
reach for most people. A used RTX 3090 costs a fraction of an H100, but the
latest models (DeepSeek V4.1, GLM, Qwen) increasingly depend on features
upstream builds for newer silicon: fp8 hardware paths, sparse attention
kernels, fused quantized cache writes.

**vllm-ampere-extended closes that gap.** We make serving the latest models
on Ampere and older hardware work — and publish every piece of software,
validation, and data it took.

## What we do

- **Ampere sparse attention.** Our `TRITON_MLA_SPARSE` backend brings DeepSeek
  sparse-MLA to SM80/SM121, with software e4m3 encode where cards lack fp8
  units — paths upstream gates to newer hardware. For these model classes
  there is no dense fallback: sparse attention is part of the trained
  architecture, so either it works on your hardware or the model doesn't run.
- **Testbed, then upstream.** Anything consistent with vLLM's mission is
  contributed upstream. What upstream doesn't take — because it serves
  hardware they don't prioritize — lives here, validated and maintained.
  This repo is the nexus for platform/feature combinations upstream doesn't
  target but that expand what common, affordable hardware can run.
- **Production-validated releases.** Every build passes a two-stage canary
  against live serving traffic before it is tagged. A tag here means the
  exact build we run in production.
- **Cherry-picked fixes.** Unmerged upstream PRs we need are staged, reviewed,
  and shipped here — you don't wait on upstream review cycles to run the
  latest models.
- **Recipes with data.** Model + GPU configs with benchmarks showing why they
  beat the alternatives — see [recipes/](recipes/).

## What's in our delta

| Area | What | Why here |
|---|---|---|
| Sparse attention | `TRITON_MLA_SPARSE` backend: SM80/SM121 sparse MLA, Lightning Indexer, paged-MQA logits kernels | Upstream targets SM90+ fp8 hardware paths |
| Quantization | Software e4m3 encode for indexer-Q, cache writes, compressor, kpool on pre-SM89 | Cards without fp8 units need software fallbacks |
| Models | DeepSeek V4.1, GLM5Next (NoPE sparse MLA), Qwen4Exp bring-up on SM80 | Latest models on older silicon |
| Bugfixes | int64 paged-logits addressing, spec-decode under PP, KV-offload races | General fixes — these go upstream first |

## Installation

Wheels (Linux x86_64, Python 3.12, CUDA 13.2) are attached to each release:

```bash
pip install https://github.com/practical-inference/vllm-ampere-extended/releases/download/<tag>/vllm-<version>.whl
```

Browse builds on the [Releases](https://github.com/practical-inference/vllm-ampere-extended/releases)
page. Every release notes the upstream vLLM commit it was built from.

## Support matrix

| | |
|---|---|
| GPUs | Ampere and newer (validated nightly on SM80/SM86) |
| CUDA | 13.2 wheels |
| Python | 3.12 |
| OS | Linux x86_64 |
| Cadence | Nightly, when the canary passes |

## Relationship to upstream

- `main` follows [vllm-project/vllm](https://github.com/vllm-project/vllm)
  `main` nightly. We carry a documented source delta — attention backends,
  hardware fallbacks, and model bring-up (see [DELTA.md](DELTA.md)) — not a
  fork of the engine: general fixes go upstream first, and upstream arrives
  here by merge, not by divergence.
- **Bugs in vLLM itself**: report
  [upstream](https://github.com/vllm-project/vllm/issues) — check whether it
  reproduces on an upstream release first; the bug template asks.
- **Issues with these builds** (regressions between tags, Ampere-specific
  behavior, wheel/install problems): open an issue here.

## Community

- [Contributing](.github/CONTRIBUTING.md)
- [Recipes](recipes/) — configs and the data behind them
- [Support channels](SUPPORT.md)
- [Security policy](SECURITY.md)

## License

Apache License 2.0, same as upstream vLLM.
