# Contributing to vllm-ampere-extended

This is a downstream distribution of [vLLM](https://github.com/vllm-project/vllm).
`main` tracks upstream nightly and carries a **documented source delta** —
sparse-attention backends and hardware fallbacks for Ampere-class GPUs, listed
in [DELTA.md](../DELTA.md). General fixes are contributed upstream first; the
delta here is what upstream doesn't ship for this hardware. Read
[README](../README.md) for the mission.

## What belongs here

- Fixes and improvements to the build, validation, and release tooling
- Documentation for this distribution (install, support matrix, policies)
- Reproductions and fixes for issues specific to published builds
- Ampere/older-GPU support that upstream doesn't ship — with the data showing
  why it belongs here rather than upstream

## What belongs upstream

Everything in `vllm/` source: features, model support, kernels, engine
behavior, API changes. Submit those to
[vllm-project/vllm](https://github.com/vllm-project/vllm) following their
[contribution guide](https://docs.vllm.ai/en/latest/contributing) — they arrive
here through the nightly merge. A PR here that touches `vllm/` source will be
redirected upstream unless it is part of a tracked upstream contribution.

## Sign-off (DCO)

Every commit must carry a `Signed-off-by:` trailer matching the author:

```bash
git commit -s
```

The email in the sign-off must be one GitHub credits to your account. If you
have "Keep my email addresses private" enabled, use your anonymized noreply
address (`<id>+<username>@users.noreply.github.com`) as both author email and
sign-off — a sign-off from a real address that the DCO check cannot credit
will block your PR.

## Pull requests

1. Fork, branch, and keep the change small.
2. Run the linters locally: `pre-commit run --all-files` (Python style is
   ruff, 88-column limit, Google-style docstrings).
3. Fill in the PR template — including the "why here and not upstream" section.
4. CI runs a lint/pre-commit check on every PR automatically.
5. AI-assisted contributions are welcome with disclosure: state that AI
   assistance was used, and a human must review and be accountable for every
   changed line.

## Issues

Use the issue templates. Bug reports against published builds should include
the wheel version and whether the issue reproduces on an upstream vLLM
release — that check routes the report to the right place and saves a round
trip.

## Review expectations

- First response on PRs and issues: within a few days.
- Small, focused PRs get the fastest review.
- Releases are cut by the validation pipeline; contributors cannot publish
  tags directly (branch and tag protection enforce this).

## Repo conventions for agent tooling

`.agents/skills/` holds the skill definitions (upstream's plus ours);
`.claude/skills/` contains only symlinks to them — never real files. This is
upstream's convention and is enforced by the `check_agent_files` pre-commit
hook, which also rejects any non-symlink entry, so personal agent config
cannot land here. Everything else under `.claude/` is gitignored.
