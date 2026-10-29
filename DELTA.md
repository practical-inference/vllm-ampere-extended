<!-- markdownlint-disable MD060 -->
# Delta vs upstream vLLM

`main` = upstream `main` (nightly merge) + the commits below: 17 source commits (74 files changed, 4849 insertions(+), 287 deletions(-)).

Diff base: upstream commit `3765000a37` — the tip this tree last merged from origin. Reproduce at this commit with:
`git log --no-merges 3765000a37..fb4a6ae46d -- vllm/ csrc/` and `git diff --stat 3765000a37..fb4a6ae46d -- vllm/ csrc/`.

Areas upstream does not ship for this hardware are what this distribution exists to carry; general fixes are contributed upstream first (see README). Regenerated automatically by the nightly integrate.

| Area | Commits |
|---|---|
| spec-decode fixes | 5 |
| general fixes | 5 |
| model bring-up (GLM5Next / Qwen4Exp / DSV41) | 2 |
| KV-cache / offload fixes | 1 |
| fork delta (squash: sparse attention (TRITON_MLA_SPARSE) + KV-cache / offload fixes) | 1 |
| fork delta (squash: sparse attention (TRITON_MLA_SPARSE) + spec-decode fixes) | 1 |
| sparse attention (TRITON_MLA_SPARSE) | 1 |
| fork delta (squash: sparse attention (TRITON_MLA_SPARSE) + SM80 / pre-SM89 fp8 fallbacks + model bring-up (GLM5Next / Qwen4Exp / DSV41) + spec-decode fixes + KV-cache / offload fixes) | 1 |

<details><summary>Full commit list</summary>

- fix(spec): mypy narrowing on the K=0 catch-up propose + test helper (6c1e67f8c1)
- style: ruff format scheduler.py eagle-hit strip (d187cac631)
- fix(spec): eagle K=0 hold admits requests — eagle hits stripped, table capped (38f18ce0f1)
- [Bugfix][Frontend] Compile GLM-4.7 strict tool grammars with any_order=True (3b2cb3575a)
- fix(spec): eagle K=0 catch-up next tokens must be row-aligned (df4ba64557)
- fix(spec): eagle K=0 resume re-growth must precede the block-id snapshot (add5d58351)
- feat(spec): eagle K=0 draft-resource lifecycle state machine + runner catch-up (A') (ae774cb74a)
- perf(kv): eagle-scoped allocation extras + synced-token introspection (A' groundwork) (643481013f)
- fix: bump pinned ruff to v0.16.4 (6ff6eddf7d)
- fix: fold the K=0 accepted-token offset in align preprocess (draft-less decode rows take the regular path) (6ea82baffd)
- fix: track upstream refactor in fork delta (4d1b910d91)
- fix: sync amd kda signature with nvidia + torch.accelerator in dedup test (5c1b5e6aa1)
- (no subject) (6d1e2280ee)
- [Bugfix][SpecDecode] Propagate rope overrides to same-checkpoint draft configs (f4545481f8)
- (no subject) (929f2f5fac)
- [Core] Pre-compute non-spec chunk indices for the glm5next KDA chunk path (2d94403b5c)
- chore: re-seed next onto upstream lineage — reapply the fork delta (6968d526b1)

</details>
