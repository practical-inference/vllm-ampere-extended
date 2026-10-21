# Security Policy

## Scope

This policy covers the artifacts this repository distributes: published wheels,
release tags, and the repository content itself.

For vulnerabilities in vLLM itself (the engine, models, kernels, API servers),
report **upstream** following the
[vLLM security policy](https://github.com/vllm-project/vllm/security/policy).
This distribution carries a documented source delta on top of upstream
(see [DELTA.md](DELTA.md) — sparse-attention backends and hardware fallbacks);
general fixes are contributed upstream first, and upstream fixes arrive here
through the nightly merge.

## Reporting a vulnerability in this distribution

Use [GitHub private vulnerability reporting](https://github.com/practical-inference/vllm-ampere-extended/security/advisories/new)
on this repository. Please include:

- The release tag and wheel filename affected
- The environment where you observed the issue
- A minimal reproduction, if possible

## What we will do

- Acknowledge reports within 7 days
- For issues in our build/publish pipeline (wrong artifact, tampered or
  mismatched wheel, broken provenance): fix and publish a corrected build
- For issues in vLLM source: coordinate with upstream and track their fix into
  the next published build

## Provenance

Every release tag points at the exact commit it was built from, and wheels are
built by the validation pipeline described in the
[README](README.md). If a wheel's contents do not match its tag, treat that as
a security issue and report it here.
