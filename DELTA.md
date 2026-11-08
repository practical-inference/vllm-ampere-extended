<!-- markdownlint-disable MD060 -->
# Delta vs upstream vLLM

`main` = upstream `main` (nightly merge) + the commits below: 41 source commits (61 files changed, 4359 insertions(+), 335 deletions(-)).

Diff base: upstream commit `0915f490c4` — the tip this tree last merged from origin. Reproduce at this commit with:
`git log --no-merges 0915f490c4..d0b447404c -- vllm/ csrc/` and `git diff --stat 0915f490c4..d0b447404c -- vllm/ csrc/`.

Areas upstream does not ship for this hardware are what this distribution exists to carry; general fixes are contributed upstream first (see README). Regenerated automatically by the nightly integrate.

| Area | Commits |
|---|---|
| general fixes | 16 |
| spec-decode fixes | 11 |
| sparse attention (TRITON_MLA_SPARSE) | 4 |
| model bring-up (GLM5Next / Qwen4Exp / DSV41) | 4 |
| KV-cache / offload fixes | 2 |
| fork delta (squash: spec-decode fixes + KV-cache / offload fixes) | 1 |
| fork delta (squash: sparse attention (TRITON_MLA_SPARSE) + KV-cache / offload fixes) | 1 |
| fork delta (squash: sparse attention (TRITON_MLA_SPARSE) + spec-decode fixes) | 1 |
| fork delta (squash: sparse attention (TRITON_MLA_SPARSE) + SM80 / pre-SM89 fp8 fallbacks + model bring-up (GLM5Next / Qwen4Exp / DSV41) + spec-decode fixes + KV-cache / offload fixes) | 1 |

<details><summary>Full commit list</summary>

- correction: f3e27e6ef2 'whitespace only' claim false — remove resurrected dead single_group_spec (d2c1a7fde9)
- fix: sparse-indexer SM80 fallback validation fixes (781 passed on SM80) (22714165ba)
- fix: align MTP draft QSA geometry with the shared top-k buffer (ef7d022c42)
- fix: drop upstream's duplicate process_weights_after_loading in qsa.py (d0a2f67817)
- [Bugfix][Frontend] Thread excludes/max_whitespace_cnt into every or-branch leaf (9c2a2bd005)
- [Bugfix][Frontend] Re-tighten any_order on the GLM-4.7 strict path (f424de6748)
- fix: restore the decode-path SM80 triton fallback in the sparse indexer (aa7e5f011b)
- fix: route the k=64/128 persistent-topk arms through the dtype wrapper after the upstream launcher split (0f03b25c08)
- fix: restore the SM80 triton fallback + DeepGEMM gate in the sparse indexer after the upstream restructure (aeec9d9df2)
- Revert "Merge branch 'perf/eagle-k0-lifecycle' into next" (9107e1fba5)
- Revert "fix(spec): eagle K=0 resume re-growth must reach the runner's block table" (183ef23a1a)
- Revert "fix(spec): eagle K=0 resume rebuilds runner rows via the resumed channel" (9b31b8d4e8)
- Revert "perf(sched): eagle K=0 hold engages only on all-decode steps" (b794cff8a5)
- Revert "perf(runner): batch the eagle K=0 stash capture to one clone set per step" (5d4cfa746a)
- Revert "perf(runner): prune the eagle K=0 stash only when the live set changes" (008d530292)
- Revert "docs: correct claims in 7e67d27777/9570e9e0f5/9b97d10d0b (the stash-clone chase)" (ca70f82b4f)
- docs: correct claims in 7e67d27777/9570e9e0f5/9b97d10d0b (the stash-clone chase) (91c751dbac)
- docs: regenerate DELTA.md for the current delta (9d2030103a)
- perf(runner): prune the eagle K=0 stash only when the live set changes (9b97d10d0b)
- perf(runner): batch the eagle K=0 stash capture to one clone set per step (9570e9e0f5)
- perf(sched): eagle K=0 hold engages only on all-decode steps (7e67d27777)
- fix(spec): eagle K=0 resume rebuilds runner rows via the resumed channel (72e2f45c01)
- fix(spec): eagle K=0 resume re-growth must reach the runner's block table (6d6c7e8f2e)
- docs: regenerate DELTA.md for the current delta (b017bd1ce9)
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
